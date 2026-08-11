"""Transkun model implementation for the audio-to-MIDI step."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class TranskunModel:
    """Load one Transkun checkpoint and reuse it for audio-to-MIDI jobs."""

    def __init__(
        self,
        checkpoint: str = "2.0",
        device: str = "auto",
        segment_hop_size: int | None = None,
        segment_size: int | None = None,
    ) -> None:
        try:
            import moduleconf
            import torch
            import transkun
            from transkun.Data import writeMidi
            from transkun.transcribe import readAudio
        except ImportError as exc:
            raise RuntimeError("Transkun is not installed; run `uv sync`") from exc

        self.device = choose_device(device)
        self.segment_hop_size = segment_hop_size
        self.segment_size = segment_size
        self.torch = torch
        self.read_audio = readAudio
        self.write_midi = writeMidi

        package_dir = Path(transkun.__file__).resolve().parent
        config_path = package_dir / "pretrained" / f"{checkpoint}.conf"
        weight_path = package_dir / "pretrained" / f"{checkpoint}.pt"
        if not config_path.is_file() or not weight_path.is_file():
            raise RuntimeError(f"Unknown Transkun checkpoint: {checkpoint}")
        config_manager = moduleconf.parseFromFile(str(config_path))
        model_class = config_manager["Model"].module.TransKun
        model_config = config_manager["Model"].config
        checkpoint_data: dict[str, Any] = torch.load(weight_path, map_location=self.device)

        self.model = model_class(conf=model_config).to(self.device)
        state_key = (
            "best_state_dict" if "best_state_dict" in checkpoint_data else "state_dict"
        )
        self.model.load_state_dict(checkpoint_data[state_key], strict=False)
        self.model.eval()

    def transcribe(self, input_path: Path, output_path: Path) -> None:
        try:
            import soxr
        except ImportError as exc:
            raise RuntimeError("Transkun requires soxr; run `uv sync`") from exc

        sample_rate, audio = self.read_audio(input_path)
        if sample_rate != self.model.fs:
            audio = soxr.resample(audio, sample_rate, self.model.fs)

        input_tensor = self.torch.from_numpy(audio).to(self.device)
        with self.torch.no_grad():
            notes = self.model.transcribe(
                input_tensor,
                stepInSecond=self.segment_hop_size,
                segmentSizeInSecond=self.segment_size,
                discardSecondHalf=False,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_midi(notes).write(output_path)

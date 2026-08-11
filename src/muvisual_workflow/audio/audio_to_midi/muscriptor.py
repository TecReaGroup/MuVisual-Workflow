"""MuScriptor model implementation for the audio-to-MIDI step."""

from __future__ import annotations

from pathlib import Path


class MuscriptorModel:
    """Load one MuScriptor model and reuse it for a batch of files."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str | None = None,
        instruments: tuple[str, ...] = ("acoustic_piano",),
    ) -> None:
        try:
            import torch
            from muscriptor import TranscriptionModel
        except ImportError as exc:
            raise RuntimeError("MuScriptor is not installed; run `uv sync`") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("MuScriptor requested CUDA, but no CUDA device is available")
        if dtype in {None, "auto"}:
            dtype = "float16" if device.startswith("cuda") else "float32"

        self.device = device
        self.dtype = dtype
        self.instruments = list(instruments)
        self.model = TranscriptionModel.load_model(model_name, device=device, dtype=dtype)

    def transcribe(self, input_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi_bytes = self.model.transcribe_to_midi(input_path, instruments=self.instruments)
        output_path.write_bytes(midi_bytes)

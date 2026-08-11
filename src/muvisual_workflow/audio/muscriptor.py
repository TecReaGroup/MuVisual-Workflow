"""MuScriptor model implementation for the audio-to-MIDI step."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from muvisual_workflow.config import load_config
from muvisual_workflow.paths import DEVELOP_DATA_DIR

SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})
DEFAULT_INPUT = DEVELOP_DATA_DIR / "stem_gated"
DEFAULT_OUTPUT = DEVELOP_DATA_DIR / "midi"


def find_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio with MuScriptor.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audio_to_midi = load_config(args.config).audio_to_midi
        config = audio_to_midi.for_instrument(args.instrument)
        if not args.input.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {args.input}")
        model = MuscriptorModel(
            args.checkpoint or (
                config.checkpoint if config.model == "muscriptor" else "large"
            ),
            args.device or config.device,
            args.dtype or config.dtype,
            config.target_instruments,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    files = find_audio_files(args.input)
    for source in files:
        destination = (args.output / source.relative_to(args.input)).with_suffix(".mid")
        if destination.exists() and not args.overwrite:
            continue
        model.transcribe(source, destination)
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Separate files in data/audio into stems using the BS-Roformer SW model.

Install the dependency first:
    python -m pip install -U audio-separator[gpu]

The model checkpoint is downloaded/cached by python-audio-separator when a
known model filename is supplied. Use --model to select the exact BS-Roformer
SW checkpoint available in your installed version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT_DIR = Path(__file__).parent / "data" / "audio"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "stem"
DEFAULT_MODEL_DIR = Path(__file__).parent / "data" / "model" / "BS-Rofo-SW"
DEFAULT_MODEL = "BS-Roformer-SW.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate audio into Vocal, Bass, Drums, Guitar, Piano and Other stems."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"Input file or directory (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"BS-Roformer SW checkpoint filename (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help=f"Directory containing the model checkpoint (default: {DEFAULT_MODEL_DIR})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    if input_path.is_dir() and not any(input_path.rglob("*")):
        raise SystemExit(f"Input directory is empty: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    if model_dir is not None and not (model_dir / args.model).is_file():
        available = sorted(path.name for path in model_dir.glob("*.ckpt"))
        detail = f" Available checkpoints: {', '.join(available)}" if available else ""
        raise SystemExit(
            f"Model checkpoint does not exist: {model_dir / args.model}.{detail} "
            "Rename the local checkpoint to the supported filename or pass --model with that filename."
        )

    try:
        from audio_separator.separator import Separator
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("audio_separator"):
            raise SystemExit(
                "Missing dependency: install it with "
                "python -m pip install -U audio-separator[gpu]"
            ) from exc
        raise

    separator_kwargs = {"output_dir": str(output_dir), "output_format": "WAV"}
    if model_dir is not None:
        separator_kwargs["model_file_dir"] = str(model_dir)

    separator = Separator(**separator_kwargs)
    print(f"Loading model: {args.model}")
    separator.load_model(model_filename=args.model)

    print(f"Separating {input_path} -> {output_dir}")
    output_files = separator.separate(str(input_path))
    print(f"Done. Generated {len(output_files)} stem file(s):")
    for output_file in output_files:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()

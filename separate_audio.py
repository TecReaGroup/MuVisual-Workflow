"""Separate files in data/audio into stems using the BS-Roformer SW model.

Install the dependency first:
    python -m pip install -U audio-separator[gpu]

The model checkpoint is downloaded/cached by python-audio-separator when a
known model filename is supplied. Use --model to select the exact BS-Roformer
SW checkpoint available in your installed version.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterator

DEFAULT_INPUT_DIR = Path(__file__).parent / "data" / "audio"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "stem"
DEFAULT_MODEL_DIR = Path(__file__).parent / "data" / "model" / "BS-Rofo-SW"
DEFAULT_MODEL = "BS-Roformer-SW.ckpt"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3"}


def _convert_m4a_to_wav(source: Path, destination: Path) -> None:
    """Decode an M4A audio stream to a libsndfile-compatible PCM WAV."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "M4A input requires FFmpeg, but ffmpeg was not found on PATH. "
            "Install FFmpeg and restart the terminal before trying again."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(source),
        "-map", "0:a:0",
        "-vn",
        "-c:a", "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        raise SystemExit(f"Could not decode M4A file {source}: {detail}") from exc


@contextmanager
def _separator_input(input_path: Path) -> Iterator[Path | list[Path]]:
    """Yield input suitable for audio-separator, transcoding M4A files temporarily."""
    if input_path.is_file():
        audio_files = [input_path]
    else:
        audio_files = sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

    m4a_files = [path for path in audio_files if path.suffix.lower() == ".m4a"]
    if not m4a_files:
        yield input_path
        return

    with TemporaryDirectory(prefix="muvisual-m4a-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        converted: dict[Path, Path] = {}
        for index, source in enumerate(m4a_files):
            destination = temporary_root / str(index) / f"{source.stem}.wav"
            print(f"Decoding M4A input: {source}")
            _convert_m4a_to_wav(source, destination)
            converted[source] = destination

        prepared = [converted.get(path, path) for path in audio_files]
        yield prepared[0] if input_path.is_file() else prepared


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
    with _separator_input(input_path) as prepared_input:
        if isinstance(prepared_input, list):
            separator_input = [str(path) for path in prepared_input]
        else:
            separator_input = str(prepared_input)
        output_files = separator.separate(separator_input)
    print(f"Done. Generated {len(output_files)} stem file(s):")
    for output_file in output_files:
        print(f"  {output_file}")


if __name__ == "__main__":
    main()

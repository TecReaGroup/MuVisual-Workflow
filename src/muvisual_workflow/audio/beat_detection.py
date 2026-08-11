"""Beat and downbeat detection with Beat This."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from muvisual_workflow.core.config import load_config
from muvisual_workflow.core.paths import DEVELOP_DATA_DIR, PROJECT_ROOT

SUPPORTED_EXTENSIONS = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"})
DEFAULT_INPUT = DEVELOP_DATA_DIR / "audio"
DEFAULT_OUTPUT = DEVELOP_DATA_DIR / "beats"
TEMP_DIR = PROJECT_ROOT / "temp"


def find_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _select_device(torch: Any, requested: str) -> tuple[str, bool]:
    if requested == "auto":
        return ("cuda", True) if torch.cuda.is_available() else ("cpu", False)
    if requested == "cpu":
        return "cpu", False
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Beat This requested CUDA, but no CUDA device is available")
        device = torch.device(requested)
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        return f"cuda:{index}", True
    raise ValueError("Device must be auto, cpu, cuda, or cuda:N")


def _decode_with_ffmpeg(audio_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(f"Could not decode {audio_path}; FFmpeg is not available on PATH")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=TEMP_DIR, delete=False) as file:
        decoded_path = Path(file.name)
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(decoded_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        decoded_path.unlink(missing_ok=True)
        detail = process.stderr.strip() or f"FFmpeg exited with code {process.returncode}"
        raise RuntimeError(f"Could not decode {audio_path}: {detail}")
    return decoded_path


class BeatDetector:
    """Load one Beat This model and reuse it for a batch."""

    def __init__(self, model_name: str, device: str = "auto", dbn: bool = False) -> None:
        try:
            import torch
            from beat_this.inference import File2Beats
        except ImportError as exc:
            raise RuntimeError("Beat This is not installed; run `uv sync`") from exc

        self.device, float16 = _select_device(torch, device)
        try:
            self.detector = File2Beats(
                checkpoint_path=model_name,
                device=self.device,
                float16=float16,
                dbn=dbn,
            )
        except ImportError as exc:
            if dbn:
                raise RuntimeError(
                    "Beat This DBN post-processing requires the madmom dependency"
                ) from exc
            raise

    def detect(self, audio_path: Path) -> tuple[Any, Any]:
        try:
            return self.detector(str(audio_path))
        except RuntimeError as exc:
            if "Could not load audio" not in str(exc):
                raise
        decoded_path = _decode_with_ffmpeg(audio_path)
        try:
            return self.detector(str(decoded_path))
        finally:
            decoded_path.unlink(missing_ok=True)

    def release(self) -> None:
        self.detector = None


def write_result(
    detector: BeatDetector,
    source: Path,
    destination: Path,
    audio_reference: str | None = None,
) -> None:
    beats, downbeats = detector.detect(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "audio": audio_reference or str(source),
                "beats": [float(value) for value in beats],
                "downbeats": [float(value) for value in downbeats],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def detect_path(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    dbn: bool,
) -> int:
    files = find_audio_files(input_path)
    if not files:
        print(f"No supported audio files found in {input_path}")
        return 0
    root = input_path if input_path.is_dir() else input_path.parent
    detector = BeatDetector(model_name, device, dbn)
    failures = 0
    for source in files:
        destination = (output_dir / source.relative_to(root)).with_suffix(".json")
        try:
            write_result(detector, source, destination)
            print(f"Wrote {destination}")
        except Exception as exc:  # keep processing remaining files
            failures += 1
            print(f"Failed {source}: {exc}", file=sys.stderr)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect beats and downbeats with Beat This.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dbn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config).beat_detection
    enabled = config.enabled if args.enabled is None else args.enabled
    if not enabled:
        print("Beat detection is disabled by configuration")
        return 0
    try:
        return 1 if detect_path(
            args.input,
            args.output,
            args.model or config.model,
            args.device or config.device,
            config.dbn if args.dbn is None else args.dbn,
        ) else 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

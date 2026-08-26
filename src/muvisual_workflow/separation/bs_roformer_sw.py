"""Separate files in data/develop/audio into stems using the BS-Roformer SW model.

Install project dependencies first:
    uv sync

The model checkpoint is downloaded/cached by python-audio-separator when a
known model filename is supplied. Use --model to select the exact BS-Roformer
SW checkpoint available in your installed version.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
from scipy.io import wavfile

from muvisual_workflow.core.config import NoiseGateConfig, load_config
from muvisual_workflow.core.logging import configure_logging, get_logger
from muvisual_workflow.core.paths import DATA_DIR, DEVELOP_DATA_DIR

USE_LOCAL_MODEL = True
LOCAL_MODEL_DIR = DATA_DIR / "model" / "BS-Roformer-SW"
LOCAL_MODEL_BUCKET = "hf://buckets/Trgroup/BS-Roformer-SW"
LOCAL_MODEL_FILES = ("BS-Roformer-SW.ckpt", "BS-Roformer-SW.yaml")
DEFAULT_INPUT_DIR = DEVELOP_DATA_DIR / "audio"
DEFAULT_OUTPUT_DIR = DEVELOP_DATA_DIR / "stem"
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3"}
logger = get_logger("separation")


def _to_float32(data: np.ndarray) -> tuple[np.ndarray, float]:
    if np.issubdtype(data.dtype, np.integer):
        peak = float(np.iinfo(data.dtype).max)
        return data.astype(np.float32) / peak, peak
    return data.astype(np.float32), 1.0


def _restore_dtype(data: np.ndarray, dtype: np.dtype, peak: float) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(data * peak), info.min, info.max).astype(dtype)
    return data.astype(dtype)


def _gate_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: NoiseGateConfig,
) -> np.ndarray:
    mono_or_stereo = audio if audio.ndim == 2 else audio[:, None]
    samples = len(mono_or_stereo)
    if samples == 0:
        return audio
    block = max(1, round(sample_rate * config.analysis_ms / 1000))
    count = (samples + block - 1) // block
    levels = np.empty(count, dtype=np.float32)
    for index in range(count):
        chunk = mono_or_stereo[index * block:min(samples, (index + 1) * block)]
        levels[index] = np.sqrt(np.mean(chunk * chunk))

    threshold = 10 ** (config.threshold_db / 20)
    attack_blocks = max(1, round(config.attack_ms / config.analysis_ms))
    hold_blocks = max(0, round(config.hold_ms / config.analysis_ms))
    release_blocks = max(1, round(config.release_ms / config.analysis_ms))
    gain = np.zeros(count, dtype=np.float32)
    current = 0.0
    hold_left = 0
    for index, level in enumerate(levels):
        if level >= threshold:
            hold_left = hold_blocks
            current = min(1.0, current + 1.0 / attack_blocks)
        elif hold_left:
            hold_left -= 1
        else:
            current = max(0.0, current - 1.0 / release_blocks)
        gain[index] = current

    centers = np.minimum(np.arange(count) * block + block / 2, samples - 1)
    sample_gain = np.interp(np.arange(samples), centers, gain).astype(np.float32)
    return audio * (sample_gain[:, None] if audio.ndim == 2 else sample_gain)


def apply_noise_gate(
    source: Path,
    destination: Path,
    config: NoiseGateConfig,
) -> Path:
    """Apply the BS-Roformer stem noise gate, or copy the stem when disabled."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not config.enabled:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

    sample_rate, raw = wavfile.read(source)
    audio, peak = _to_float32(raw)
    gated = _gate_audio(audio, sample_rate, config)
    wavfile.write(destination, sample_rate, _restore_dtype(gated, raw.dtype, peak))
    return destination


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
            logger.info("Decoding M4A input: %s", source)
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
    parser.add_argument("--model", default=None,
                        help="Override the separation model selected by the workflow file")
    parser.add_argument("--config", type=Path, default=None,
                        help="Instrument workflow YAML file (default: config/workflow_piano.yaml)")
    return parser.parse_args()


def prepare_local_model() -> Path:
    """Ensure the configured BS-Roformer model directory is ready to load."""
    model_dir = Path(os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", LOCAL_MODEL_DIR))
    model_dir.mkdir(parents=True, exist_ok=True)

    if not any(model_dir.iterdir()):
        try:
            from huggingface_hub import sync_bucket
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency huggingface-hub; run `uv sync` from the project root"
            ) from exc

        logger.info("Downloading local BS-Roformer model to: %s", model_dir)
        sync_bucket(
            LOCAL_MODEL_BUCKET,
            str(model_dir),
            include=list(LOCAL_MODEL_FILES),
        )

    missing_files = [name for name in LOCAL_MODEL_FILES if not (model_dir / name).is_file()]
    if missing_files:
        raise RuntimeError(
            "Local model directory is not empty but is missing required file(s): "
            f"{', '.join(missing_files)}. Directory: {model_dir}"
        )
    return model_dir


def create_separator(
    output_dir: Path,
    model: str,
    output_single_stem: str | None = None,
) -> Any:
    try:
        from audio_separator.separator import Separator
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("audio_separator"):
            raise SystemExit(
                "Missing dependency: run `uv sync` from the project root"
            ) from exc
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    separator_kwargs = {
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "output_single_stem": output_single_stem,
    }
    model_dir = prepare_local_model() if USE_LOCAL_MODEL else None
    if model_dir:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        separator_kwargs["model_file_dir"] = str(model_dir)

    separator = Separator(**separator_kwargs)
    if model_dir:
        logger.info("Loading model from local directory: %s", model_dir / model)
    else:
        logger.info("Loading model from audio-separator registry: %s", model)
    separator.load_model(model_filename=model)
    model_instance = separator.model_instance
    if output_single_stem is not None and hasattr(model_instance, "process_all_stems"):
        instruments = list(model_instance.model_data_cfgdict.training.instruments)
        matching_stem = next(
            (stem for stem in instruments if stem.casefold() == output_single_stem.casefold()),
            None,
        )
        if matching_stem is None:
            raise SystemExit(
                f"Stem {output_single_stem!r} is not available. Available stems: "
                + ", ".join(instruments)
            )
        model_instance.process_all_stems = False
        model_instance.primary_stem_name = matching_stem
        if model_instance.secondary_stem_name == matching_stem:
            model_instance.secondary_stem_name = next(
                stem for stem in instruments if stem != matching_stem
            )
    return separator


def separate_with_loaded_model(separator: Any, input_path: Path, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    separator.output_dir = str(output_dir)
    separator.model_instance.output_dir = str(output_dir)

    logger.info("Separating %s -> %s", input_path, output_dir)
    with _separator_input(input_path) as prepared_input:
        if isinstance(prepared_input, list):
            separator_input = [str(path) for path in prepared_input]
        else:
            separator_input = str(prepared_input)
        output_files = separator.separate(separator_input)
    return output_files


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config).require_separation()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")
    if input_path.is_dir() and not any(input_path.rglob("*")):
        raise SystemExit(f"Input directory is empty: {input_path}")

    separator = create_separator(output_dir, args.model or config.model)
    output_files = separate_with_loaded_model(separator, input_path, output_dir)
    if config.noise_gate.enabled:
        for output_file in output_files:
            output_path = Path(output_file)
            if not output_path.is_absolute():
                output_path = output_dir / output_path
            apply_noise_gate(output_path, output_path, config.noise_gate)
        logger.info("Applied noise gate to %d stem file(s)", len(output_files))
    logger.info("Done. Generated %d stem file(s):", len(output_files))
    for output_file in output_files:
        logger.info("Output: %s", output_file)


if __name__ == "__main__":
    main()

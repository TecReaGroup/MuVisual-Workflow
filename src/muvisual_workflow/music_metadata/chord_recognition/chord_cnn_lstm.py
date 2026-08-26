"""Chord recognition with a locally installed Chord-CNN-LSTM model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from muvisual_workflow.core.config import ChordRecognitionConfig
from muvisual_workflow.core.paths import PROJECT_ROOT

INFERENCE_SCRIPT = Path("chord_recognition.py")
CHECKPOINTS = tuple(Path("cache_data") / f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{index}.best.sdict" for index in range(5))
LEGACY_RUNNER = Path(__file__).with_name("legacy.py")


@dataclass(frozen=True)
class ChordSegment:
    start: float
    end: float
    chord: str


def resolve_repository(config: ChordRecognitionConfig) -> Path:
    repository = config.repository_path.expanduser()
    if not repository.is_absolute():
        repository = PROJECT_ROOT / repository
    repository = repository.resolve()
    required = (
        repository / INFERENCE_SCRIPT,
        repository / "data" / f"{config.chord_dictionary}_chord_list.txt",
        *(repository / checkpoint for checkpoint in CHECKPOINTS),
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Chord-CNN-LSTM model is not prepared under "
            f"{repository}: "
            + ", ".join(str(path.relative_to(repository)) for path in missing)
        )
    return repository


def read_segments(path: Path) -> list[ChordSegment]:
    segments: list[ChordSegment] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            start, end, chord = line.split(maxsplit=2)
            segments.append(ChordSegment(float(start), float(end), chord))
        except ValueError as exc:
            raise RuntimeError(f"Invalid chord LAB line {line_number}: {line!r}") from exc
    return segments


def align_to_beats(segments: list[ChordSegment], beats: list[float]) -> list[dict[str, int | float | str]]:
    if not segments or not beats:
        return []
    assignments: dict[int, str] = {}
    beat_index = 0
    for segment in segments:
        duration = beats[beat_index + 1] - beats[beat_index] if beat_index < len(beats) - 1 else beats[beat_index] - beats[beat_index - 1] if beat_index else 0.0
        while beat_index < len(beats) - 1 and beats[beat_index + 1] - duration * 0.5 <= segment.start:
            beat_index += 1
        assignments[beat_index] = segment.chord
    result: list[dict[str, int | float | str]] = []
    current = "N"
    for index, timestamp in enumerate(beats):
        current = assignments.get(index, current)
        result.append({"beat": index + 1, "time": timestamp, "chord": current})
    return result


def prepare_audio(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Chord-CNN-LSTM requires FFmpeg, but ffmpeg was not found on PATH"
        )
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = process.stderr.strip() or f"FFmpeg exited with code {process.returncode}"
        raise RuntimeError(f"Could not prepare chord analysis audio: {detail}")


def recognize_chords(audio_path: Path, beats: list[float], config: ChordRecognitionConfig) -> dict[str, object]:
    repository = resolve_repository(config)
    environment = os.environ.copy()
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    with TemporaryDirectory(prefix="muvisual-chord-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        prepared_audio_path = temporary_path / f"{audio_path.stem}.wav"
        lab_path = temporary_path / f"{audio_path.stem}.lab"
        prepare_audio(audio_path, prepared_audio_path)
        subprocess.run(
            [sys.executable, str(LEGACY_RUNNER), str(repository / INFERENCE_SCRIPT), str(prepared_audio_path.resolve()), str(lab_path.resolve()), config.chord_dictionary],
            cwd=repository,
            check=True,
            env=environment,
        )
        if not lab_path.is_file():
            raise RuntimeError(f"Chord-CNN-LSTM did not create output: {lab_path}")
        segments = read_segments(lab_path)
    return {
        "model": "Chord CNN-LSTM ISMIR 2019",
        "dictionary": config.chord_dictionary,
        "alignment": "nearest_beat_midpoint",
        "segments": [asdict(segment) for segment in segments],
        "beat_chords": align_to_beats(segments, beats),
    }

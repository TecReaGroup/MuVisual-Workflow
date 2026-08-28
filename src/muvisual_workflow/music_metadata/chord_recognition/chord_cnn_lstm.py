"""Chord recognition with a locally installed Chord-CNN-LSTM model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from muvisual_workflow.core.config import ChordRecognitionConfig
from muvisual_workflow.core.logging import get_logger
from muvisual_workflow.core.paths import PROJECT_ROOT
from muvisual_workflow.music_metadata.key_bpm_delay import (
    estimate_key_from_pitch_weights,
)

INFERENCE_SCRIPT = Path("chord_recognition.py")
CHECKPOINTS = tuple(Path("cache_data") / f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{index}.best.sdict" for index in range(5))
LEGACY_RUNNER = Path(__file__).with_name("legacy.py")
REPOSITORY_URL = "https://github.com/kuchin/chord-cnn-lstm-model.git"
logger = get_logger("music_metadata.chord_recognition")
PITCH_CLASSES = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
NATURAL_PITCH_CLASSES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
NOTE_LETTERS = ("C", "D", "E", "F", "G", "A", "B")
CHORD_QUALITY_SYMBOLS = {
    "maj": "",
    "min": "m",
    "maj7": "maj7",
    "min7": "m7",
    "7": "7",
    "dim": "dim",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "aug": "aug",
    "9": "9",
    "maj9": "maj9",
    "min9": "m9",
    "11": "11",
    "13": "13",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus4(b7)": "7sus4",
}
BASS_INTERVALS = {
    "2": (2, 2),
    "b3": (3, 3),
    "3": (3, 4),
    "5": (5, 7),
    "b7": (7, 10),
}


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
    if not missing:
        return repository
    if repository.exists():
        raise RuntimeError(
            "Chord-CNN-LSTM model directory is incomplete under "
            f"{repository}: "
            + ", ".join(str(path.relative_to(repository)) for path in missing)
        )

    repository.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            "Chord-CNN-LSTM is not prepared and git was not found on PATH: "
            f"{repository}"
        )
    logger.info("Downloading Chord-CNN-LSTM model repository: %s", repository)
    try:
        subprocess.run(
            [git, "clone", "--depth", "1", REPOSITORY_URL, str(repository)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        if repository.is_dir():
            shutil.rmtree(repository)
        raise RuntimeError(
            "Could not download Chord-CNN-LSTM model repository: "
            f"{REPOSITORY_URL}"
        ) from exc

    missing = [path for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Downloaded Chord-CNN-LSTM repository is incomplete under "
            f"{repository}: "
            + ", ".join(str(path.relative_to(repository)) for path in missing)
        )
    logger.info("Downloaded Chord-CNN-LSTM model repository: %s", repository)
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


def interval_bass_note(root: str, interval: str) -> str:
    """Convert a scale-degree inversion into a conventionally spelled bass note."""
    degree, semitones = BASS_INTERVALS[interval]
    root_letter = root[0]
    bass_letter = NOTE_LETTERS[(NOTE_LETTERS.index(root_letter) + degree - 1) % 7]
    bass_pitch = (PITCH_CLASSES[root] + semitones) % 12
    accidental = (bass_pitch - NATURAL_PITCH_CLASSES[bass_letter]) % 12
    if accidental > 6:
        accidental -= 12
    suffix = "#" * accidental if accidental > 0 else "b" * -accidental
    return f"{bass_letter}{suffix}"


def normalize_chord_symbol(chord: str) -> str:
    """Convert the model's Harte-style label to common chord-symbol notation."""
    if chord == "N" or ":" not in chord:
        return chord
    root, quality = chord.split(":", maxsplit=1)
    quality_name, separator, bass_interval = quality.partition("/")
    quality_symbol = CHORD_QUALITY_SYMBOLS.get(quality_name, quality_name)
    normalized = f"{root}{quality_symbol}"
    if separator:
        normalized += f"/{interval_bass_note(root, bass_interval)}"
    return normalized


def normalize_segments(segments: list[ChordSegment]) -> list[ChordSegment]:
    """Normalize every recognized chord before exposing the final payload."""
    return [
        replace(segment, chord=normalize_chord_symbol(segment.chord))
        for segment in segments
    ]


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


def estimate_key(segments: list[ChordSegment]) -> str:
    """Estimate the global key from duration-weighted recognized chords."""
    pitch_weights = [0.0] * 12
    for segment in segments:
        if segment.chord == "N":
            continue
        root_name, _, quality = segment.chord.partition(":")
        root = PITCH_CLASSES.get(root_name)
        if root is None:
            continue
        duration = max(0.0, segment.end - segment.start)
        if not duration:
            continue
        if quality.startswith(("min", "dim", "hdim")):
            third = 3
        elif quality.startswith(("sus2", "sus4")):
            third = 2 if quality.startswith("sus2") else 5
        else:
            third = 4
        fifth = (
            6
            if quality.startswith(("dim", "hdim"))
            else 8
            if quality.startswith("aug")
            else 7
        )
        for interval, strength in ((0, 1.0), (third, 0.8), (fifth, 0.7)):
            pitch_weights[(root + interval) % 12] += duration * strength
    return estimate_key_from_pitch_weights(pitch_weights)


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
        raw_segments = read_segments(lab_path)
    key = estimate_key(raw_segments)
    segments = normalize_segments(raw_segments)
    return {
        "model": "Chord CNN-LSTM ISMIR 2019",
        "dictionary": config.chord_dictionary,
        "key": key,
        "alignment": "nearest_beat_midpoint",
        "segments": [asdict(segment) for segment in segments],
        "beat_chords": align_to_beats(segments, beats),
    }

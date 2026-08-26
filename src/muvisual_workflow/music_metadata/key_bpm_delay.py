"""Analyze MIDI files and store song-level metadata as JSON without modifying MIDI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

try:
    import mido
except ImportError:
    mido = None  # type: ignore[assignment]

from muvisual_workflow.core.config import load_config
from muvisual_workflow.core.logging import configure_logging, get_logger
from muvisual_workflow.core.paths import DEVELOP_DATA_DIR

DEFAULT_INPUT = DEVELOP_DATA_DIR / "midi"
DEFAULT_OUTPUT = DEVELOP_DATA_DIR / "metadata"
ALIGNMENT_SAMPLE_COUNT = 35
logger = get_logger("music_metadata")

MAJOR_PROFILE = (6.35, 2.18, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def corr(a: list[float], b: tuple[float, ...]) -> float:
    am = sum(a) / 12
    bm = sum(b) / 12
    num = sum((x - am) * (y - bm) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - am) ** 2 for x in a))
    den_b = math.sqrt(sum((y - bm) ** 2 for y in b))
    return num / (den_a * den_b) if den_a and den_b else 0.0


def estimate_key(mid: mido.MidiFile) -> str:
    weights = [0.0] * 12
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity:
                weights[msg.note % 12] += msg.velocity
    if not any(weights):
        return "Unknown"
    scores = []
    for root in range(12):
        rotated = [weights[(root + i) % 12] for i in range(12)]
        scores.append((corr(rotated, MAJOR_PROFILE), f"{NAMES[root]} major"))
        scores.append((corr(rotated, MINOR_PROFILE), f"{NAMES[root]} minor"))
    return max(scores, key=lambda item: item[0])[1]

def tempo_map(mid: mido.MidiFile) -> list[tuple[int, int]]:
    events = [(0, mido.bpm2tempo(120))]
    absolute = 0
    for msg in mido.merge_tracks(mid.tracks):
        absolute += msg.time
        if msg.type == "set_tempo":
            events.append((absolute, msg.tempo))
    return sorted(events)


def tick_to_seconds(tick: int, ticks_per_beat: int, tempos: list[tuple[int, int]]) -> float:
    seconds = 0.0
    previous_tick = 0
    tempo = tempos[0][1]
    for change_tick, new_tempo in tempos[1:]:
        if change_tick >= tick:
            break
        seconds += mido.tick2second(change_tick - previous_tick, ticks_per_beat, tempo)
        previous_tick = change_tick
        tempo = new_tempo
    return seconds + mido.tick2second(tick - previous_tick, ticks_per_beat, tempo)


def estimate_bpm(onsets: list[float]) -> float:
    """Estimate a global pulse using a harmonic interval-voting histogram."""
    unique = sorted(set(round(value, 4) for value in onsets))
    if len(unique) < 4:
        return 120.0
    scores: dict[float, float] = {}
    for index, start in enumerate(unique):
        for end in unique[index + 1:index + 9]:
            interval = end - start
            if interval > 4.0:
                break
            if interval < 0.12:
                continue
            bpm = 60.0 / interval
            while bpm < 50:
                bpm *= 2
            while bpm > 200:
                bpm /= 2
            bucket = round(bpm * 2) / 2
            scores[bucket] = scores.get(bucket, 0.0) + 1.0 / math.sqrt(interval)
    if not scores:
        return 120.0
    best = max(scores, key=scores.get)
    nearby = [(bpm, score) for bpm, score in scores.items() if abs(bpm - best) <= 2]
    return sum(bpm * score for bpm, score in nearby) / sum(score for _, score in nearby)


def estimate_delay(
    notes: list[tuple[float, int, float]],
    beat_seconds: float,
    alignment_sample_count: int = ALIGNMENT_SAMPLE_COUNT,
) -> tuple[float, int, int, float]:
    """Find the positive time offset of the BPM grid, without moving MIDI.

    With ``grid_delay`` as the returned value, the positive/negative beat grid
    is ``grid_delay + k * (beat_seconds / 2)``. Each onset is scored by its
    distance to the closest grid line before or after it.
    """
    ordered_onsets = sorted(notes)
    minimum_duration = beat_seconds * 1.7 / 4.0
    samples: list[tuple[float, int]] = []
    for onset, velocity, duration in ordered_onsets:
        if duration >= minimum_duration:
            samples.append((onset, velocity))
        if len(samples) == alignment_sample_count:
            break
    if not samples:
        return 0.0, 0, 0, 0.0

    half_beat = beat_seconds / 2.0

    def alignment_scores(grid_delay: float) -> tuple[float, int, float]:
        grid_error = 0.0
        positive_beat_count = 0
        accent_error = 0.0
        for onset, velocity in samples:
            phase = (onset - grid_delay) % half_beat
            distance_to_previous = phase
            distance_to_next = half_beat - phase
            grid_error += min(distance_to_previous, distance_to_next)

            nearest_grid_index = round((onset - grid_delay) / half_beat)
            if nearest_grid_index % 2 == 0:
                positive_beat_count += 1
            else:
                beat_phase = (onset - grid_delay) % beat_seconds
                distance_to_previous_beat = beat_phase
                distance_to_next_beat = beat_seconds - beat_phase
                accent_error += min(distance_to_previous_beat, distance_to_next_beat) * velocity
        return grid_error, positive_beat_count, accent_error

    resolution = 0.0001
    candidate_count = math.floor(half_beat / resolution)
    base_grid_delay = 0.0
    base_grid_error = alignment_scores(base_grid_delay)[0]
    for index in range(1, candidate_count + 1):
        grid_delay = index * resolution
        grid_error = alignment_scores(grid_delay)[0]
        if grid_error < base_grid_error:
            base_grid_delay = grid_delay
            base_grid_error = grid_error

    mirrored_delays = (base_grid_delay, base_grid_delay + half_beat)
    mirrored_scores = {
        grid_delay: alignment_scores(grid_delay)
        for grid_delay in mirrored_delays
    }
    best_grid_delay = min(
        mirrored_delays,
        key=lambda grid_delay: (
            -mirrored_scores[grid_delay][1],
            mirrored_scores[grid_delay][2],
            grid_delay,
        ),
    )
    best_grid_error, best_positive_count, _ = mirrored_scores[best_grid_delay]
    return best_grid_delay, best_positive_count, len(samples), best_grid_error


@dataclass(frozen=True)
class MusicMetadata:
    """Metadata inferred from one instrument MIDI file."""

    key: str
    bpm: float
    delay: float
    positive_count: int
    sample_count: int
    alignment_error: float

    def to_dict(self) -> dict[str, object]:
        negative_count = self.sample_count - self.positive_count
        average_error = (
            self.alignment_error / self.sample_count if self.sample_count else 0.0
        )
        return {
            "key": self.key,
            "bpm": self.bpm,
            "delay": self.delay,
            "alignment": {
                "positive_count": self.positive_count,
                "negative_count": negative_count,
                "sample_count": self.sample_count,
                "error": self.alignment_error,
                "average_error": average_error,
            },
        }


def analyze_file(
    source: Path,
    alignment_sample_count: int = ALIGNMENT_SAMPLE_COUNT,
) -> MusicMetadata:
    """Infer musical metadata without changing or saving the MIDI file."""
    mid = mido.MidiFile(source)
    key = estimate_key(mid)
    tempos = tempo_map(mid)
    onset_seconds: list[float] = []
    note_events: list[tuple[float, int, float]] = []
    for track in mid.tracks:
        absolute = 0
        active_notes: dict[tuple[int, int], list[tuple[float, int]]] = {}
        for msg in track:
            absolute += msg.time
            if msg.type == "note_on" and msg.velocity:
                onset = tick_to_seconds(absolute, mid.ticks_per_beat, tempos)
                onset_seconds.append(onset)
                note_key = (msg.channel, msg.note)
                active_notes.setdefault(note_key, []).append((onset, msg.velocity))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                note_key = (msg.channel, msg.note)
                starts = active_notes.get(note_key)
                if starts:
                    onset, velocity = starts.pop(0)
                    release = tick_to_seconds(absolute, mid.ticks_per_beat, tempos)
                    note_events.append((onset, velocity, max(0.0, release - onset)))

    bpm = estimate_bpm(onset_seconds)
    beat_seconds = 60.0 / bpm
    delay, positive_count, sample_count, alignment_error = estimate_delay(
        note_events,
        beat_seconds,
        alignment_sample_count,
    )
    return MusicMetadata(
        key=key,
        bpm=bpm,
        delay=delay,
        positive_count=positive_count,
        sample_count=sample_count,
        alignment_error=alignment_error,
    )


def load_song_metadata(path: Path) -> dict[str, object]:
    """Load an existing song metadata object so instrument workflows can merge."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read metadata JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Metadata JSON root must be an object: {path}")
    return payload


def update_song_metadata(
    path: Path,
    *,
    audio: str | None = None,
    beats: list[float] | None = None,
    downbeats: list[float] | None = None,
    beat_metadata: Mapping[str, object] | None = None,
    chords: Mapping[str, object] | None = None,
    instrument: str | None = None,
    instrument_metadata: MusicMetadata | None = None,
) -> None:
    """Merge beat or instrument analysis into ``歌名_meta.json``."""
    payload = load_song_metadata(path)
    if audio is not None:
        payload["audio"] = audio
    if beats is not None:
        payload["beats"] = beats
    if downbeats is not None:
        payload["downbeats"] = downbeats
    if beat_metadata is not None:
        payload["beat"] = dict(beat_metadata)
    if chords is not None:
        payload["chords"] = dict(chords)
    if instrument is not None:
        if instrument_metadata is None:
            raise ValueError("instrument_metadata is required when instrument is set")
        instruments = payload.setdefault("instruments", {})
        if not isinstance(instruments, dict):
            raise RuntimeError(f"Metadata field 'instruments' must be an object: {path}")
        instruments[instrument] = instrument_metadata.to_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Estimate MIDI key, BPM, and grid delay without modifying MIDI."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--instrument", default="unknown")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config).require_music_metadata()
    if mido is None:
        raise SystemExit("Missing dependency: run `uv sync` from the project root")

    input_path = args.input.expanduser().resolve()
    files = (
        [input_path]
        if input_path.is_file()
        else sorted(input_path.glob("*.mid")) + sorted(input_path.glob("*.midi"))
    )
    if not files:
        raise SystemExit(f"No MIDI files found in {input_path}")

    output_path = args.output.expanduser().resolve()
    for source in files:
        metadata = analyze_file(source, config.key_bpm_delay.alignment_sample_count)
        destination = (
            output_path
            if len(files) == 1 and output_path.suffix.lower() == ".json"
            else output_path / f"{source.stem}_meta.json"
        )
        update_song_metadata(
            destination,
            instrument=args.instrument,
            instrument_metadata=metadata,
        )
        logger.info(
            "%s: key=%s, bpm=%.2f, delay=%.1fms -> %s",
            source.name,
            metadata.key,
            metadata.bpm,
            metadata.delay * 1000,
            destination,
        )


if __name__ == "__main__":
    main()

"""Analyze and gently normalize MIDI files.

Dependencies:
    python -m pip install mido

For each MIDI file this script reports an estimated key and BPM, removes tempo
changes, writes one global tempo event, and applies one uniform time shift to
the complete performance. Individual notes are never quantized or moved
relative to one another.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import mido
except ImportError:
    mido = None  # type: ignore[assignment]


ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "data" / "midi"
DEFAULT_OUTPUT = ROOT / "data" / "midi_fixed"

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

def midi_key_signature(key_name: str) -> str | None:
    """Convert a display name such as C# major to Mido key name."""
    if key_name == "Unknown":
        return None
    root, mode = key_name.rsplit(" ", 1)
    return f"{root}m" if mode == "minor" else root


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


def normalize_track(
    track: mido.MidiTrack,
    ticks_per_beat: int,
    tempos: list[tuple[int, int]],
    new_tempo: int,
    shift_seconds: float,
) -> mido.MidiTrack:
    events: list[tuple[int, mido.Message]] = []
    absolute = 0
    for msg in track:
        absolute += msg.time
        seconds = tick_to_seconds(absolute, ticks_per_beat, tempos) + shift_seconds
        new_tick = round(mido.second2tick(max(0.0, seconds), ticks_per_beat, new_tempo))
        events.append((new_tick, msg.copy()))

    normalized = mido.MidiTrack()
    previous = 0
    for tick, msg in events:
        msg.time = max(0, tick - previous)
        previous = tick
        normalized.append(msg)
    return normalized


def remove_tempo_events(track: mido.MidiTrack) -> None:
    carried_time = 0
    messages = []
    for msg in track:
        if msg.type == "set_tempo":
            carried_time += msg.time
            continue
        copied = msg.copy(time=msg.time + carried_time)
        carried_time = 0
        messages.append(copied)
    track[:] = messages

def remove_key_signature_events(track: mido.MidiTrack) -> None:
    carried_time = 0
    messages = []
    for msg in track:
        if msg.type == "key_signature":
            carried_time += msg.time
            continue
        messages.append(msg.copy(time=msg.time + carried_time))
        carried_time = 0
    track[:] = messages


def normalize_file(source: Path, destination: Path) -> tuple[str, float]:
    mid = mido.MidiFile(source)
    key = estimate_key(mid)
    tempos = tempo_map(mid)
    onset_seconds = []
    for track in mid.tracks:
        absolute = 0
        for msg in track:
            absolute += msg.time
            if msg.type == "note_on" and msg.velocity:
                onset_seconds.append(tick_to_seconds(absolute, mid.ticks_per_beat, tempos))
    bpm = estimate_bpm(onset_seconds)
    new_tempo = mido.bpm2tempo(bpm)
    first_onset = min(onset_seconds, default=0.0)
    beat_seconds = 60.0 / bpm
    remainder = first_onset % beat_seconds
    shift_seconds = -remainder if remainder <= beat_seconds / 2 else beat_seconds - remainder
    tracks = [
        normalize_track(track, mid.ticks_per_beat, tempos, new_tempo, shift_seconds)
        for track in mid.tracks
    ]
    # Keep one global tempo at the beginning of the first track.
    for track in tracks:
        remove_tempo_events(track)
        remove_key_signature_events(track)
    tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=new_tempo, time=0))
    key_for_midi = midi_key_signature(key)
    if key_for_midi is not None:
        tracks[0].insert(1, mido.MetaMessage("key_signature", key=key_for_midi, time=0))
    mid.tracks = tracks
    destination.parent.mkdir(parents=True, exist_ok=True)
    mid.save(destination)
    return key, bpm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate MIDI key/BPM, write one tempo, and apply one global timing offset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if mido is None:
        raise SystemExit("Missing dependency. Install with: python -m pip install mido")
    files = sorted(args.input.glob("*.mid")) + sorted(args.input.glob("*.midi"))
    if not files:
        raise SystemExit(f"No MIDI files found in {args.input.resolve()}")
    for source in files:
        destination = args.output / source.name
        key, bpm = normalize_file(source, destination)
        print(f"{source.name}: key={key}, bpm={bpm:.2f} -> {destination}")


if __name__ == "__main__":
    main()

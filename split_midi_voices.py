"""Assign Partitura/VoSA piano hand labels to MIDI channels.

Install:
    python -m pip install partitura mido

The input directory is expected to contain MIDI files. Each output keeps the
original events, but note-on/note-off events are moved to two hand channels:
left hand (channel 1) and right hand (channel 2).
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "midi_fixed"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "midi_hand_split"
MONOPHONIC_VOICES = False
DEBUG = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate MIDI notes into left/right hand channels using Partitura analysis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--monophonic-voices",
        action="store_true",
        default=MONOPHONIC_VOICES,
        help="Force each estimated voice to be monophonic.",
    )
    return parser.parse_args()


def get_field(row, name: str, fallback):
    return row[name].item() if name in row.dtype.names else fallback


def build_tempo_map(mid):
    import mido

    changes = [(0, mido.bpm2tempo(120))]
    absolute = 0
    for message in mido.merge_tracks(mid.tracks):
        absolute += message.time
        if message.type == "set_tempo":
            changes.append((absolute, message.tempo))
    return changes


def tick_to_seconds(tick: int, ppq: int, tempo_map) -> float:
    import mido

    seconds = 0.0
    previous = 0
    tempo = tempo_map[0][1]
    for change_tick, new_tempo in tempo_map[1:]:
        if change_tick >= tick:
            break
        seconds += mido.tick2second(change_tick - previous, ppq, tempo)
        previous = change_tick
        tempo = new_tempo
    return seconds + mido.tick2second(tick - previous, ppq, tempo)


def estimate_voice_map(source: Path, mid, monophonic: bool) -> dict[tuple[int, int], int]:
    import numpy as np
    import partitura as pt
    from partitura.musicanalysis import estimate_voices

    performance = pt.load_performance_midi(source, merge_tracks=False, quiet=True)
    note_array = performance.note_array()
    if len(note_array) == 0:
        return {}
    estimate_voices(note_array, monophonic_voices=monophonic)

    tempo_map = build_tempo_map(mid)
    note_events = []
    for track_index, track in enumerate(mid.tracks):
        absolute = 0
        for message_index, message in enumerate(track):
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                note_events.append({
                    "event": (track_index, message_index),
                    "track": track_index,
                    "channel": message.channel,
                    "pitch": message.note,
                    "onset_sec": tick_to_seconds(absolute, mid.ticks_per_beat, tempo_map),
                })

    grouped_events = defaultdict(list)
    for event in note_events:
        grouped_events[
            (event["track"], event["channel"], event["pitch"])
        ].append(event)
    for events in grouped_events.values():
        events.sort(key=lambda item: item["onset_sec"])

    grouped_notes = defaultdict(list)
    for index, row in enumerate(note_array):
        track = int(get_field(row, "track", 0))
        channel = int(get_field(row, "channel", 0))
        pitch = int(row["pitch"])
        onset = float(get_field(row, "onset_sec", 0.0))
        grouped_notes[(track, channel, pitch)].append((onset, index))
    for notes in grouped_notes.values():
        notes.sort(key=lambda item: item[0])

    voice_values = np.asarray(voices)
    matched_voices = {}
    for key, notes in grouped_notes.items():
        events = grouped_events.get(key, [])
        # Partitura may renumber tracks while loading a performance. If the
        # track number does not match Mido's track index, fall back to the
        # stable channel/pitch/onset grouping.
        if not events:
            loose_events = [
                event for event in note_events
                if event["channel"] == key[1] and event["pitch"] == key[2]
            ]
            if not loose_events:
                loose_events = [
                    event for event in note_events if event["pitch"] == key[2]
                ]
            loose_events.sort(key=lambda item: item["onset_sec"])
        events = loose_events
        for event, (_, note_index) in zip(events, notes):
            matched_voices[event["event"]] = int(voice_values[note_index])

    # VoSA returns independent voices, not hands. Cluster the voices by their
    # own median register, so the split adapts to the piece instead of using
    # a fixed C4 boundary.
    voice_pitches = defaultdict(list)
    event_pitches = {event["event"]: event["pitch"] for event in note_events}
    for event_key, voice in matched_voices.items():
        voice_pitches[voice].append(event_pitches[event_key])
    voice_registers = {
        voice: float(np.median(pitches))
        for voice, pitches in voice_pitches.items()
    }
    voice_ids = list(voice_registers)
    if len(voice_ids) < 2:
        return {event_key: 1 for event_key in matched_voices}

    centers = np.array([
        min(voice_registers.values()),
        max(voice_registers.values()),
    ], dtype=float)
    for _ in range(16):
        assignments = {
            voice: int(np.argmin(np.abs(centers - register)))
            for voice, register in voice_registers.items()
        }
        new_centers = np.array([
            np.mean([
                register for voice, register in voice_registers.items()
                if assignments[voice] == cluster
            ]) if any(assignments[voice] == cluster for voice in voice_registers)
            else centers[cluster]
            for cluster in (0, 1)
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    left_cluster = int(np.argmin(centers))
    return {
        event_key: 1 if assignments[voice] == left_cluster else 2
        for event_key, voice in matched_voices.items()
    }


def apply_voice_channels(mid, voice_map: dict[tuple[int, int], int]) -> None:
    """Apply voice channels while pairing note-off events with note-on events."""
    active = defaultdict(deque)
    for track_index, track in enumerate(mid.tracks):
        for message_index, message in enumerate(track):
            is_on = message.type == "note_on" and message.velocity > 0
            is_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )
            if is_on:
                voice = voice_map.get((track_index, message_index))
                if voice is None:
                    continue
                channel = (voice - 1) % 16
                active[(track_index, message.channel, message.note)].append(channel)
                message.channel = channel
            elif is_off:
                key = (track_index, message.channel, message.note)
                if active[key]:
                    message.channel = active[key].popleft()


def channel_note_counts(mid) -> dict[int, int]:
    counts = defaultdict(int)
    for track in mid.tracks:
        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                counts[message.channel] += 1
    return dict(sorted(counts.items()))


def process_file(source: Path, destination: Path, monophonic: bool) -> tuple[int, dict[int, int], dict[int, int]]:
    import mido

    mid = mido.MidiFile(source)
    before_counts = channel_note_counts(mid)
    voice_map = estimate_voice_map(source, mid, monophonic)
    apply_voice_channels(mid, voice_map)
    after_counts = channel_note_counts(mid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mid.save(destination)
    saved = mido.MidiFile(destination)
    saved_counts = channel_note_counts(saved)
    if DEBUG and after_counts != saved_counts:
        raise RuntimeError(
            f"Saved MIDI channel verification failed: memory={after_counts}, file={saved_counts}"
        )
    return len(set(voice_map.values())), before_counts, saved_counts


def main() -> None:
    args = parse_args()
    files = sorted(args.input.glob("*.mid")) + sorted(args.input.glob("*.midi"))
    if not files:
        raise SystemExit(f"No MIDI files found in {args.input.resolve()}")
    try:
        import mido  # noqa: F401
        import partitura  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python -m pip install partitura mido"
        ) from exc

    for source in files:
        destination = args.output / source.name
        voice_count, before_counts, saved_counts = process_file(
            source, destination, args.monophonic_voices
        )
        print(f"{source.name}: assigned {voice_count} hand channel(s)")
        if DEBUG:
            print(f"  original channels: {before_counts or 'none'}")
            print(f"  saved channels:    {saved_counts or 'none'}")
            for channel, count in saved_counts.items():
                print(f"    MIDI channel {channel + 1}: {count} note(s)")
        print(f"  output: {destination}")


if __name__ == "__main__":
    main()

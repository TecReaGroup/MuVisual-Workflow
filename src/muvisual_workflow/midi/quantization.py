"""Post-process generated MIDI note timing."""

from __future__ import annotations

from pathlib import Path


def quantize_midi(source: Path, destination: Path | None = None) -> Path:
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("Mido is not installed; run `uv sync`") from exc

    midi = mido.MidiFile(source)
    c4_note = 60
    time_threshold = 100

    for track in midi.tracks:
        if not any(message.type in {"note_on", "note_off"} for message in track):
            continue

        notes_on = []
        notes_off = []
        other_events = []
        current_time = 0
        for message in track:
            current_time += message.time
            if message.type == "note_on" and message.velocity > 0:
                notes_on.append({
                    "time": current_time,
                    "note": message.note,
                    "velocity": message.velocity,
                    "channel": message.channel,
                })
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                notes_off.append({
                    "time": current_time,
                    "note": message.note,
                    "velocity": message.velocity if message.type == "note_off" else 0,
                    "channel": message.channel,
                })
            else:
                other_events.append({"time": current_time, "msg": message})

        def process_hand_notes(hand_notes: list[dict]) -> None:
            hand_notes.sort(key=lambda note: note["time"])
            index = 0
            while index < len(hand_notes):
                current = hand_notes[index]["time"]
                simultaneous = [hand_notes[index]]
                next_index = index + 1
                while (
                    next_index < len(hand_notes)
                    and hand_notes[next_index]["time"] - current <= time_threshold
                ):
                    simultaneous.append(hand_notes[next_index])
                    next_index += 1

                if next_index < len(hand_notes):
                    next_time = hand_notes[next_index]["time"]
                    for note in simultaneous:
                        best_off = None
                        best_distance = float("inf")
                        for off in notes_off:
                            if (
                                off["note"] == note["note"]
                                and off["channel"] == note["channel"]
                                and off["time"] > note["time"]
                                and not off.get("processed", False)
                            ):
                                distance = off["time"] - note["time"]
                                if distance < best_distance:
                                    best_distance = distance
                                    best_off = off
                        if best_off is not None:
                            best_off["time"] = max(note["time"] + 100, next_time - 10)
                            best_off["processed"] = True
                index = next_index

        process_hand_notes([note for note in notes_on if note["note"] <= c4_note])
        process_hand_notes([note for note in notes_on if note["note"] > c4_note])

        all_events = (
            [("note_on", note["time"], note) for note in notes_on]
            + [("note_off", note["time"], note) for note in notes_off]
            + [("other", event["time"], event) for event in other_events]
        )
        all_events.sort(key=lambda event: event[1])

        rebuilt = mido.MidiTrack()
        last_time = 0
        for event_type, event_time, event_data in all_events:
            delta = event_time - last_time
            if event_type == "note_on":
                message = mido.Message(
                    "note_on",
                    channel=event_data["channel"],
                    note=event_data["note"],
                    velocity=event_data["velocity"],
                    time=delta,
                )
            elif event_type == "note_off":
                message = mido.Message(
                    "note_off",
                    channel=event_data["channel"],
                    note=event_data["note"],
                    velocity=event_data["velocity"],
                    time=delta,
                )
            else:
                message = event_data["msg"].copy(time=delta)
            rebuilt.append(message)
            last_time = event_time
        track.clear()
        track.extend(rebuilt)

    output = destination or source.with_name(f"{source.stem}_quantized{source.suffix}")
    midi.save(output)
    return output

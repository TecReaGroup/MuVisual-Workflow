"""Post-process generated MIDI note timing."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

from muvisual_workflow.core.config import MidiQuantizationConfig, load_config
from muvisual_workflow.core.logging import configure_logging, get_logger
from muvisual_workflow.core.paths import DEVELOP_DATA_DIR


DEFAULT_INPUT = DEVELOP_DATA_DIR / "midi_fixed"
DEFAULT_AUDIO = DEVELOP_DATA_DIR / "stem_gated"
DEFAULT_OUTPUT = DEVELOP_DATA_DIR / "midi_quantized"
logger = get_logger("midi_quantization")


def quantize_midi(
    source: Path,
    destination: Path | None = None,
    audio_path: Path | None = None,
    config: MidiQuantizationConfig | None = None,
) -> Path:
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("Mido is not installed; run `uv sync`") from exc

    if audio_path is None:
        raise RuntimeError("Quantization requires the corresponding audio file")
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                raise RuntimeError(f"Invalid WAV sample rate: {audio_path}")
            audio_duration_seconds = audio.getnframes() / frame_rate
    except (OSError, wave.Error) as exc:
        raise RuntimeError(f"Could not read WAV duration: {audio_path}: {exc}") from exc

    settings = config or MidiQuantizationConfig()
    midi = mido.MidiFile(source)
    tempo = next(
        (
            message.tempo
            for track in midi.tracks
            for message in track
            if message.type == "set_tempo"
        ),
        500_000,
    )
    audio_end_tick = round(
        mido.second2tick(audio_duration_seconds, midi.ticks_per_beat, tempo)
    )
    c4_note = settings.hand_split_note
    time_threshold = settings.simultaneous_threshold_ticks

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

                is_last_group = next_index >= len(hand_notes)
                next_time = (
                    None if is_last_group else hand_notes[next_index]["time"]
                )
                for note in simultaneous:
                    if is_last_group and note["time"] >= audio_end_tick:
                        raise RuntimeError(
                            f"Final MIDI note starts at or after the audio end: {source}"
                        )
                    target_end = (
                        audio_end_tick
                        if is_last_group
                        else max(
                            note["time"] + settings.minimum_note_ticks,
                            next_time - settings.next_group_gap_ticks,
                        )
                    )
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
                    if best_off is None:
                        best_off = {
                            "time": target_end,
                            "note": note["note"],
                            "velocity": 0,
                            "channel": note["channel"],
                        }
                        notes_off.append(best_off)
                    else:
                        best_off["time"] = target_end
                    best_off["processed"] = True
                index = next_index

        process_hand_notes([note for note in notes_on if note["note"] <= c4_note])
        process_hand_notes([note for note in notes_on if note["note"] > c4_note])

        for event in other_events:
            if event["msg"].type == "end_of_track":
                event["time"] = audio_end_tick

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
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output)
    return output


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Quantize normalized MIDI files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config).require_midi_quantization()

    input_path = args.input.expanduser().resolve()
    audio_path = args.audio.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in {".mid", ".midi"}:
            raise SystemExit(f"Input must be a MIDI file: {input_path}")
        destination = (
            output_path
            if output_path.suffix.lower() in {".mid", ".midi"}
            else output_path / input_path.name
        )
        source_audio = (
            audio_path
            if audio_path.is_file()
            else audio_path / f"{input_path.stem}.wav"
        )
        logger.info(
            "Quantized: %s -> %s",
            input_path,
            quantize_midi(input_path, destination, source_audio, config),
        )
        return

    if not input_path.is_dir():
        raise SystemExit(f"Input MIDI path does not exist: {input_path}")

    files = sorted(input_path.glob("*.mid")) + sorted(input_path.glob("*.midi"))
    if not files:
        raise SystemExit(f"No MIDI files found in {input_path}")
    for source in files:
        destination = output_path / source.name
        source_audio = audio_path / f"{source.stem}.wav"
        logger.info(
            "Quantized: %s -> %s",
            source,
            quantize_midi(source, destination, source_audio, config),
        )


if __name__ == "__main__":
    main()

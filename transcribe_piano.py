"""Transcribe the separated piano stem to MIDI with Transkun.

Install Transkun first:
    python -m pip install transkun

The Transkun command downloads/loads its pretrained weights automatically.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "stem_gated" / "一生爱你_(piano)_BS-Roformer-SW.wav"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "midi" / "一生爱你_(piano)_BS-Roformer-SW.mid"
ENABLE_QUANTIZE = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a piano stem to MIDI with Transkun.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input WAV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output MIDI (default: {DEFAULT_OUTPUT})")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: auto; uses CUDA when PyTorch reports it available)",
    )
    parser.add_argument("--segment-hop-size", type=int, default=None, help="Optional Transkun segmentHopSize")
    parser.add_argument("--segment-size", type=int, default=None, help="Optional Transkun segmentSize")
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def midi_quantize(source: Path) -> Path:
    try:
        import mido
    except ImportError as exc:
        raise SystemExit("Quantization requires: python -m pip install mido") from exc

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

    output = source.with_name(f"{source.stem}_quantized{source.suffix}")
    midi.save(output)
    return output


def trim_midi_silence(midi) -> None:
    first_note_time = float("inf")
    last_note_time = 0
    for track in midi.tracks:
        current_time = 0
        has_note = False
        for message in track:
            current_time += message.time
            if message.type == "note_on" and message.velocity > 0:
                first_note_time = min(first_note_time, current_time)
                last_note_time = max(last_note_time, current_time)
                has_note = True
    if first_note_time == float("inf"):
        return

    for track in midi.tracks:
        current_time = 0
        kept = []
        for message in track:
            current_time += message.time
            if (
                first_note_time <= current_time <= last_note_time + 1000
                or message.type in {"set_tempo", "key_signature", "time_signature"}
                or current_time < first_note_time
            ):
                kept.append((max(0, current_time - first_note_time), message))
        track.clear()
        previous = 0
        for absolute_time, message in kept:
            track.append(message.copy(time=absolute_time - previous))
            previous = absolute_time


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input audio does not exist: {input_path}")

    transkun = shutil.which("transkun")
    if transkun is None:
        raise SystemExit(
            "Transkun command not found. Install it with: python -m pip install transkun"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    command = [transkun, str(input_path), str(output_path), "--device", device]
    if args.segment_hop_size is not None:
        command.extend(["--segmentHopSize", str(args.segment_hop_size)])
    if args.segment_size is not None:
        command.extend(["--segmentSize", str(args.segment_size)])

    print(f"Transcribing: {input_path}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("Unable to start Transkun. Check the installation in this Python environment.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Transkun failed with exit code {exc.returncode}") from exc

    if not output_path.is_file():
        raise SystemExit(f"Transkun completed but did not create: {output_path}")
    print(f"Done: {output_path}")
    if ENABLE_QUANTIZE:
        quantized_path = midi_quantize(output_path)
        print(f"Quantized copy: {quantized_path}")


if __name__ == "__main__":
    main()

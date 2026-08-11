"""Run the configurable audio-to-MIDI workflow step.

Install project dependencies first:
    uv sync

MuScriptor and Transkun are selectable model implementations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Protocol

from muvisual_workflow.config import InstrumentAudioToMidiConfig, load_config
from muvisual_workflow.paths import DEVELOP_DATA_DIR

DEFAULT_INPUT = DEVELOP_DATA_DIR / "stem_gated"
DEFAULT_OUTPUT = DEVELOP_DATA_DIR / "midi"
INSTRUMENT_LABEL = re.compile(r"(?:^|[_\s-])\(([^)]+)\)(?=[_\s.-]|$)", re.IGNORECASE)


class AudioToMidiModel(Protocol):
    device: str
    model: object

    def transcribe(self, input_path: Path, output_path: Path) -> None: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a piano stem to MIDI.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input WAV file or folder (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output MIDI file or folder (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Override the configured inference device",
    )
    parser.add_argument("--model", choices=("muscriptor", "transkun"), default=None)
    parser.add_argument("--instrument", default=None, help="Override the instrument name")
    parser.add_argument("--checkpoint", default=None, help="Override the model checkpoint")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--segment-hop-size", type=int, default=None, help="Optional Transkun segmentHopSize")
    parser.add_argument("--segment-size", type=int, default=None, help="Optional Transkun segmentSize")
    return parser.parse_args()


def create_audio_to_midi_model(config: InstrumentAudioToMidiConfig) -> AudioToMidiModel:
    """Construct the selected model implementation for this workflow step."""
    if config.model == "muscriptor":
        from muvisual_workflow.audio.muscriptor import MuscriptorModel

        return MuscriptorModel(
            model_name=config.checkpoint,
            device=config.device,
            dtype=config.dtype,
            instruments=config.target_instruments,
        )
    from muvisual_workflow.audio.transkun import TranskunModel

    return TranskunModel(
        checkpoint=config.checkpoint,
        device=config.device,
        segment_hop_size=config.segment_hop_size,
        segment_size=config.segment_size,
    )


def midi_quantize(source: Path) -> Path:
    try:
        import mido
    except ImportError as exc:
        raise SystemExit("Missing Mido dependency: run `uv sync` from the project root") from exc

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


@dataclass(frozen=True)
class AudioToMidiResult:
    midi_path: Path
    quantized_path: Path | None


class AudioToMidiStep:
    """Execute one configured audio-to-MIDI model and optional quantization."""

    def __init__(self, config: InstrumentAudioToMidiConfig) -> None:
        self.config = config
        self.model = create_audio_to_midi_model(config)

    @property
    def device(self) -> str:
        return self.model.device

    def run(self, input_path: Path, output_path: Path) -> AudioToMidiResult:
        print(f"Audio-to-MIDI model: {self.config.model} ({self.config.checkpoint})")
        print(f"Device: {self.device}")
        print(f"Transcribing: {input_path}")
        self.model.transcribe(input_path, output_path)
        if not output_path.is_file():
            raise RuntimeError(f"Audio-to-MIDI did not create: {output_path}")

        quantized_path = midi_quantize(output_path) if self.config.quantize else None
        print(f"Wrote: {output_path}")
        if quantized_path is not None:
            print(f"Quantized copy: {quantized_path}")
        return AudioToMidiResult(output_path, quantized_path)

    def release(self) -> None:
        """Release the loaded implementation model."""
        if hasattr(self.model, "model"):
            self.model.model = None


def detect_instrument(audio_path: Path) -> str | None:
    match = INSTRUMENT_LABEL.search(audio_path.name)
    return match.group(1).casefold() if match else None


def main() -> None:
    args = parse_args()
    configured = load_config(args.config).audio_to_midi
    input_path = args.input.expanduser().resolve()
    configured_output = args.output.expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise SystemExit(f"Input audio must be a WAV file: {input_path}")
        output_path = (
            configured_output
            if configured_output.suffix.lower() == ".mid"
            else configured_output / f"{input_path.stem}.mid"
        )
        instrument = args.instrument or detect_instrument(input_path) or configured.default_instrument
        inputs = [(input_path, output_path, instrument.casefold())]
    elif input_path.is_dir():
        inputs = []
        for audio_path in sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".wav"
        ):
            relative_path = audio_path.relative_to(input_path).with_suffix(".mid")
            instrument = args.instrument or detect_instrument(audio_path)
            if instrument is None:
                instrument = configured.default_instrument
            inputs.append((audio_path, configured_output / relative_path, instrument.casefold()))
        if not inputs:
            raise SystemExit(f"Input folder contains no WAV files: {input_path}")
    else:
        raise SystemExit(f"Input audio does not exist: {input_path}")

    steps: dict[str, AudioToMidiStep] = {}
    for audio_path, output_path, instrument in inputs:
        if instrument not in steps:
            try:
                instrument_config = configured.for_instrument(instrument)
            except ValueError as exc:
                print(f"Skipping {audio_path}: {exc}")
                continue
            selected_model = args.model or instrument_config.model
            selected_checkpoint = args.checkpoint
            if selected_checkpoint is None:
                selected_checkpoint = (
                    "large"
                    if args.model == "muscriptor"
                    else "2.0" if args.model == "transkun" else instrument_config.checkpoint
                )
            instrument_config = replace(
                instrument_config,
                model=selected_model,
                checkpoint=selected_checkpoint,
                device=args.device or instrument_config.device,
                dtype=(
                    instrument_config.dtype
                    if args.dtype is None
                    else None if args.dtype == "auto" else args.dtype
                ),
                segment_hop_size=(
                    args.segment_hop_size
                    if args.segment_hop_size is not None
                    else instrument_config.segment_hop_size
                ),
                segment_size=(
                    args.segment_size
                    if args.segment_size is not None
                    else instrument_config.segment_size
                ),
            )
            steps[instrument] = AudioToMidiStep(instrument_config)
        step = steps[instrument]
        print(f"Instrument: {instrument}")
        step.run(audio_path, output_path)


if __name__ == "__main__":
    main()

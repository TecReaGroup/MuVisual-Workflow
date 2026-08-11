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

from muvisual_workflow.core.config import InstrumentAudioToMidiConfig, load_config
from muvisual_workflow.core.paths import DEVELOP_DATA_DIR
from muvisual_workflow.midi.quantization import quantize_midi

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
        from muvisual_workflow.audio.audio_to_midi.muscriptor import MuscriptorModel

        return MuscriptorModel(
            model_name=config.checkpoint,
            device=config.device,
            dtype=config.dtype,
            instruments=config.target_instruments,
        )
    from muvisual_workflow.audio.audio_to_midi.transkun import TranskunModel

    return TranskunModel(
        checkpoint=config.checkpoint,
        device=config.device,
        segment_hop_size=config.segment_hop_size,
        segment_size=config.segment_size,
    )


@dataclass(frozen=True)
class AudioToMidiResult:
    midi_path: Path
    quantized: bool


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

        quantized = self.config.model == "transkun"
        if quantized:
            quantize_midi(output_path, output_path)
        print(f"Wrote: {output_path}")
        if quantized:
            print(f"Quantized in place: {output_path}")
        return AudioToMidiResult(output_path, quantized)

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

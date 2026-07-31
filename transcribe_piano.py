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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also write a 1/16-note quantized copy with a _quantized suffix. "
             "The original MIDI is kept unchanged; pedal timing may be affected.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def write_quantized_copy(source: Path) -> Path:
    try:
        import mido
    except ImportError as exc:
        raise SystemExit("Quantization requires: python -m pip install mido") from exc

    midi = mido.MidiFile(source)
    grid = max(1, round(midi.ticks_per_beat / 4))
    quantized_tracks = []
    for track in midi.tracks:
        absolute = 0
        events = []
        for message in track:
            absolute += message.time
            target = absolute
            if message.type in {"note_on", "note_off"}:
                target = max(0, round(absolute / grid) * grid)
            events.append((target, message.copy()))
        output_track = mido.MidiTrack()
        previous = 0
        for tick, message in events:
            message.time = max(0, tick - previous)
            previous = tick
            output_track.append(message)
        quantized_tracks.append(output_track)
    midi.tracks = quantized_tracks
    output = source.with_name(f"{source.stem}_quantized{source.suffix}")
    midi.save(output)
    return output


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
    if args.quantize:
        quantized_path = write_quantized_copy(output_path)
        print(f"Quantized copy: {quantized_path}")


if __name__ == "__main__":
    main()

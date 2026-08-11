"""Assign piano MIDI notes to left- and right-hand channels with Piano_SVSep.

Install the model repository and its dependencies first:
    git clone https://github.com/CPJKU/piano_svsep.git
    cd piano_svsep
    uv pip install .

Set PIANO_SVSEP_ROOT below if the repository is stored elsewhere. The
pretrained model is expected at pretrained_models/model.ckpt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import mido

from muvisual_workflow.core.paths import DATA_DIR, DEVELOP_DATA_DIR, PROJECT_ROOT

DEFAULT_INPUT = DEVELOP_DATA_DIR / "midi_fixed"
DEFAULT_OUTPUT = DATA_DIR / "midi_hand_split"
PIANO_SVSEP_ROOT = Path(
    os.environ.get("PIANO_SVSEP_ROOT", str(PROJECT_ROOT / "piano_svsep"))
)
MODEL_PATH = PIANO_SVSEP_ROOT / "pretrained_models" / "model.ckpt"
DEBUG = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Piano_SVSep staff predictions to write left/right MIDI channels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--piano-svsep-root", type=Path, default=PIANO_SVSEP_ROOT)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def channel_note_counts(mid: mido.MidiFile) -> dict[int, int]:
    counts = defaultdict(int)
    for track in mid.tracks:
        for message in track:
            if message.type == "note_on" and message.velocity > 0:
                counts[message.channel] += 1
    return dict(sorted(counts.items()))


def original_note_events(mid: mido.MidiFile) -> list[dict]:
    events = []
    for track_index, track in enumerate(mid.tracks):
        absolute = 0
        for message_index, message in enumerate(track):
            absolute += message.time
            if message.type == "note_on" and message.velocity > 0:
                events.append({
                    "event": (track_index, message_index),
                    "track": track_index,
                    "channel": message.channel,
                    "pitch": message.note,
                    "tick": absolute,
                })
    return events


def predict_staff(source: Path, mid: mido.MidiFile, repo_root: Path, model: Path) -> dict[tuple[int, int], int]:
    """Run the official Piano_SVSep predictor and map staff labels to MIDI notes."""
    try:
        import partitura as pt
    except ImportError as exc:
        raise SystemExit(
            "Missing Partitura. Install Piano_SVSep with: uv pip install ."
        ) from exc

    predict_script = repo_root / "launch_scripts" / "predict.py"
    if not predict_script.is_file():
        raise SystemExit(f"Piano_SVSep predictor not found: {predict_script}")
    if not model.is_file():
        raise SystemExit(f"Piano_SVSep model not found: {model}")

    with tempfile.TemporaryDirectory(prefix="piano_svsep_") as temp:
        temp_dir = Path(temp)
        score_input = temp_dir / "input.musicxml"
        score_output = temp_dir / "predicted.mei"

        score = pt.load_score_midi(
            source,
            assign_note_ids=True,
            quiet=True,
        )
        pt.save_musicxml(score, score_input)
        subprocess.run(
            [
                os.fspath(__import__("sys").executable),
                os.fspath(predict_script),
                "--model_path",
                os.fspath(model),
                "--score_path",
                os.fspath(score_input),
                "--save_path",
                os.fspath(score_output),
            ],
            cwd=os.fspath(repo_root),
            check=True,
        )

        predicted_score = pt.load_score(score_output, force_note_ids=True)
        predicted_array = predicted_score.note_array(include_staff=True)
        if "staff" not in (predicted_array.dtype.names or ()):
            raise RuntimeError("Piano_SVSep output does not contain staff labels")

        predicted_groups = defaultdict(list)
        onset_name = "onset_beat" if "onset_beat" in predicted_array.dtype.names else "onset_div"
        for row in predicted_array:
            onset = float(row[onset_name])
            tick = round(onset * mid.ticks_per_beat)
            predicted_groups[(int(row["pitch"]), tick)].append(int(row["staff"]))

        source_groups = defaultdict(list)
        for event in original_note_events(mid):
            source_groups[(event["pitch"], event["tick"])].append(event)

        staff_map = {}
        for key, events in source_groups.items():
            staffs = predicted_groups.get(key, [])
            for event, staff in zip(events, staffs):
                # Piano_SVSep predicts two staves. Map staff 1 to channel 1
                # and every other staff label to channel 2.
                staff_map[event["event"]] = 1 if staff == 1 else 2
        return staff_map


def apply_hand_channels(mid: mido.MidiFile, hand_map: dict[tuple[int, int], int]) -> None:
    active = defaultdict(deque)
    for track_index, track in enumerate(mid.tracks):
        for message_index, message in enumerate(track):
            is_on = message.type == "note_on" and message.velocity > 0
            is_off = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )
            if is_on:
                hand = hand_map.get((track_index, message_index))
                if hand is None:
                    continue
                old_channel = message.channel
                new_channel = hand - 1
                active[(track_index, old_channel, message.note)].append(new_channel)
                message.channel = new_channel
            elif is_off:
                key = (track_index, message.channel, message.note)
                if active[key]:
                    message.channel = active[key].popleft()


def process_file(source: Path, destination: Path, repo_root: Path, model: Path) -> None:
    mid = mido.MidiFile(source)
    before = channel_note_counts(mid)
    hand_map = predict_staff(source, mid, repo_root, model)
    apply_hand_channels(mid, hand_map)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mid.save(destination)

    saved = mido.MidiFile(destination)
    after = channel_note_counts(saved)
    if DEBUG:
        print(f"{source.name}: matched {len(hand_map)} note(s)")
        print(f"  original channels: {before or 'none'}")
        print(f"  saved channels:    {after or 'none'}")
        for channel, count in after.items():
            print(f"    MIDI channel {channel + 1}: {count} note(s)")
        if sum(before.values()) != sum(after.values()):
            print("  WARNING: note count changed during staff mapping")
        print(f"  output: {destination}")


def main() -> None:
    args = parse_args()
    files = sorted(args.input.glob("*.mid")) + sorted(args.input.glob("*.midi"))
    if not files:
        raise SystemExit(f"No MIDI files found in {args.input.resolve()}")
    repo_root = args.piano_svsep_root.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"Piano_SVSep repository does not exist: {repo_root}")
    for source in files:
        process_file(source, args.output / source.name, repo_root, model)


if __name__ == "__main__":
    main()

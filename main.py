from __future__ import annotations

import argparse
import gc
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mutagen

from separate_audio import (
    AUDIO_EXTENSIONS,
    DEFAULT_MODEL,
    DEFAULT_MODEL_DIR,
    create_separator,
    separate_with_loaded_model,
)
from transcribe_piano import TranskunTranscriber, midi_quantize

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "data" / "input"
DEFAULT_OUTPUT = ROOT / "data" / "output"
INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class PreparedAudio:
    source: Path
    output_name: str
    destination_dir: Path
    result_dir: Path
    gated_piano: Path
    midi_dir: Path
    fixed_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process every audio file through the MuVisual pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--segment-hop-size", type=int, default=None)
    parser.add_argument("--segment-size", type=int, default=None)
    return parser.parse_args()


def run_script(script: str, *arguments: object) -> None:
    command = [sys.executable, str(ROOT / script), *(str(argument) for argument in arguments)]
    subprocess.run(command, check=True)


def clean_filename_part(value: str, tag_name: str, source: Path) -> str:
    cleaned = INVALID_FILENAME_CHARACTERS.sub("_", value).strip(" .")
    if not cleaned:
        raise RuntimeError(f"Audio tag {tag_name!r} is empty: {source}")
    return cleaned


def read_output_name(source: Path) -> str:
    try:
        audio = mutagen.File(source, easy=True)
    except mutagen.MutagenError as exc:
        raise RuntimeError(f"Could not read audio tags: {source}: {exc}") from exc
    if audio is None:
        raise RuntimeError(f"Mutagen could not identify the audio format: {source}")

    tags = audio.tags
    if tags is None:
        raise RuntimeError(f"Audio has no metadata tags: {source}")

    values: dict[str, str] = {}
    for tag_name in ("title", "album"):
        tag_values = tags.get(tag_name)
        if not tag_values or not str(tag_values[0]).strip():
            raise RuntimeError(f"Audio is missing the {tag_name!r} tag: {source}")
        values[tag_name] = clean_filename_part(str(tag_values[0]).strip(), tag_name, source)
    return f"{values['title']}_{values['album']}"


def find_piano_stem(stem_dir: Path) -> Path:
    piano_label = re.compile(r"(?:^|[_\s-])\(piano\)(?=[_\s.-]|$)", re.IGNORECASE)
    candidates = sorted(
        path
        for path in stem_dir.glob("*.wav")
        if piano_label.search(path.name)
    )
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(f"Expected one piano stem, found {len(candidates)}: {names}")
    return candidates[0]


def convert_to_mp3(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".mp3":
        shutil.copy2(source, destination)
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found on PATH; it is required to create MP3 output")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-i", str(source),
            "-map", "0:a:0",
            "-vn",
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            str(destination),
        ],
        check=True,
    )


def prepare_audio(
    source: Path,
    output_name: str,
    output_root: Path,
    work_dir: Path,
    separator: object,
) -> PreparedAudio:
    destination_dir = output_root / output_name
    stem_dir = work_dir / "stem"
    piano_stem_dir = work_dir / "piano_stem"
    gated_dir = work_dir / "stem_gated"
    midi_dir = work_dir / "midi"
    fixed_dir = work_dir / "midi_fixed"
    result_dir = work_dir / "result"
    result_dir.mkdir(parents=True)

    original_mp3 = result_dir / f"{output_name}.mp3"
    piano_mp3 = result_dir / f"{output_name}_piano.mp3"
    convert_to_mp3(source, original_mp3)
    output_files = separate_with_loaded_model(separator, source, stem_dir)
    print(f"Generated {len(output_files)} stem file(s): {', '.join(output_files)}")
    piano_stem = find_piano_stem(stem_dir)
    piano_stem_dir.mkdir()
    shutil.copy2(piano_stem, piano_stem_dir / piano_stem.name)
    run_script("gate_stems.py", "--input", piano_stem_dir, "--output", gated_dir)
    gated_piano = gated_dir / piano_stem.name
    if not gated_piano.is_file():
        raise RuntimeError(f"Gated piano stem was not created: {gated_piano}")
    convert_to_mp3(gated_piano, piano_mp3)

    return PreparedAudio(
        source=source,
        output_name=output_name,
        destination_dir=destination_dir,
        result_dir=result_dir,
        gated_piano=gated_piano,
        midi_dir=midi_dir,
        fixed_dir=fixed_dir,
    )


def finish_audio(job: PreparedAudio, transcriber: TranskunTranscriber) -> None:
    midi_path = job.midi_dir / f"{job.output_name}.mid"
    quantized_path = job.midi_dir / f"{job.output_name}_quantized.mid"

    print(f"Transcribing: {job.gated_piano}")
    transcriber.transcribe(job.gated_piano, midi_path)
    midi_quantize(midi_path)
    if not midi_path.is_file() or not quantized_path.is_file():
        raise RuntimeError("Transcription did not create both MIDI files")

    run_script("fix_midi.py", "--input", job.midi_dir, "--output", job.fixed_dir)
    fixed_midi = job.fixed_dir / midi_path.name
    fixed_quantized = job.fixed_dir / quantized_path.name
    if not fixed_midi.is_file() or not fixed_quantized.is_file():
        raise RuntimeError("MIDI fix did not create both output files")

    job.destination_dir.mkdir(parents=True, exist_ok=True)
    results = (
        job.result_dir / f"{job.output_name}.mp3",
        job.result_dir / f"{job.output_name}_piano.mp3",
        fixed_midi,
        fixed_quantized,
    )
    for result in results:
        shutil.copy2(result, job.destination_dir / result.name)


def release_separator(separator: object) -> None:
    model_instance = getattr(separator, "model_instance", None)
    if hasattr(separator, "model_instance"):
        separator.model_instance = None
    del model_instance
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    args.model_dir = args.model_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    audio_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise SystemExit(f"No audio files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, str]] = []
    named_audio_files: list[tuple[Path, str]] = []
    for source in audio_files:
        try:
            named_audio_files.append((source, read_output_name(source)))
        except RuntimeError as exc:
            failures.append((source, str(exc)))

    name_counts = Counter(output_name.casefold() for _, output_name in named_audio_files)
    unique_audio_files: list[tuple[Path, str]] = []
    for source, output_name in named_audio_files:
        if name_counts[output_name.casefold()] > 1:
            failures.append(
                (source, f"Duplicate title/album output name: {output_name}")
            )
        else:
            unique_audio_files.append((source, output_name))

    with TemporaryDirectory(prefix="muvisual-batch-") as temporary_dir:
        batch_work_dir = Path(temporary_dir)
        separator = create_separator(
            batch_work_dir / "separator",
            args.model,
            args.model_dir,
            output_single_stem="Piano",
        )
        prepared_jobs: list[PreparedAudio] = []
        for index, (source, output_name) in enumerate(unique_audio_files, start=1):
            print(f"\n[Separate {index}/{len(unique_audio_files)}] Processing: {source}")
            try:
                job = prepare_audio(
                    source,
                    output_name,
                    output_dir,
                    batch_work_dir / str(index),
                    separator,
                )
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                failures.append((source, str(exc)))
                print(f"Failed: {source}: {exc}", file=sys.stderr)
            else:
                prepared_jobs.append(job)

        release_separator(separator)
        del separator

        if prepared_jobs:
            transcriber = TranskunTranscriber(
                device=args.device,
                segment_hop_size=args.segment_hop_size,
                segment_size=args.segment_size,
            )
            for index, job in enumerate(prepared_jobs, start=1):
                print(
                    f"\n[Transcribe {index}/{len(prepared_jobs)}] Processing: "
                    f"{job.source}"
                )
                try:
                    finish_audio(job, transcriber)
                except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                    failures.append((job.source, str(exc)))
                    print(f"Failed: {job.source}: {exc}", file=sys.stderr)
                else:
                    print(f"Completed: {job.destination_dir}")

    if failures:
        details = "\n".join(f"  {source}: {error}" for source, error in failures)
        raise SystemExit(f"\n{len(failures)} of {len(audio_files)} file(s) failed:\n{details}")
    print(f"\nProcessed {len(audio_files)} file(s) successfully.")


if __name__ == "__main__":
    main()

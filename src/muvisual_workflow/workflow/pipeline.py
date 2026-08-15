from __future__ import annotations

import argparse
import gc
import re
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import mutagen

from muvisual_workflow.audio.audio_to_midi import AudioToMidiStep
from muvisual_workflow.audio.beat_detection import BeatDetector, write_result
from muvisual_workflow.audio.conversion import convert_to_mp3
from muvisual_workflow.audio.noise_gate import gate_file
from muvisual_workflow.audio.separation import (
    AUDIO_EXTENSIONS,
    create_separator,
    separate_with_loaded_model,
)
from muvisual_workflow.core.config import (
    AudioToMidiConfig,
    BeatDetectionConfig,
    InstrumentAudioToMidiConfig,
    load_config,
)
from muvisual_workflow.core.paths import DATA_DIR, PROJECT_ROOT
from muvisual_workflow.midi.normalizer import normalize_file
from muvisual_workflow.midi.quantization import quantize_midi

DEFAULT_INPUT = DATA_DIR / "input"
DEFAULT_OUTPUT = DATA_DIR / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
STEM_LABEL = re.compile(r"\(([^)]+)\)(?=[_\s.-]|$)", re.IGNORECASE)
BS_ROFORMER_SW_STEMS = ("bass", "drums", "guitar", "other", "piano", "vocals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process every audio file through the MuVisual pipeline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        "--separation-model",
        dest="model",
        default=None,
        help="Override the configured separation model",
    )
    parser.add_argument(
        "--audio-to-midi-model",
        choices=("muscriptor", "transkun"),
        default=None,
    )
    parser.add_argument("--audio-to-midi-checkpoint", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help="Override every configured audio-to-MIDI device",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML configuration file (default: config/muvisual.toml)",
    )
    parser.add_argument("--segment-hop-size", type=int, default=None)
    parser.add_argument("--segment-size", type=int, default=None)
    return parser.parse_args()


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
    if audio.tags is None:
        raise RuntimeError(f"Audio has no metadata tags: {source}")

    values: dict[str, str] = {}
    for tag_name in ("title", "album"):
        tag_values = audio.tags.get(tag_name)
        if not tag_values or not str(tag_values[0]).strip():
            raise RuntimeError(f"Audio is missing the {tag_name!r} tag: {source}")
        values[tag_name] = clean_filename_part(
            str(tag_values[0]).strip(),
            tag_name,
            source,
        )
    return f"{values['title']}_{values['album']}"


def discover_instrument_stems(stem_dir: Path) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for path in sorted(stem_dir.rglob("*.wav")):
        match = STEM_LABEL.search(path.name)
        if match is None:
            raise RuntimeError(f"Could not identify the instrument for stem: {path.name}")
        instrument = match.group(1).strip().casefold()
        if not instrument:
            raise RuntimeError(f"Stem has an empty instrument label: {path.name}")
        if instrument in stems:
            raise RuntimeError(
                f"Found multiple {instrument!r} stems: "
                f"{stems[instrument].name}, {path.name}"
            )
        stems[instrument] = path
    if not stems:
        raise RuntimeError(f"Separation did not create any WAV stems in: {stem_dir}")
    return stems


def expected_model_stems(model: str) -> tuple[str, ...] | None:
    if model.casefold() == "bs-roformer-sw.ckpt":
        return BS_ROFORMER_SW_STEMS
    return None


def resolve_instrument_configs(
    config: AudioToMidiConfig,
    args: argparse.Namespace,
) -> dict[str, InstrumentAudioToMidiConfig]:
    resolved: dict[str, InstrumentAudioToMidiConfig] = {}
    for instrument, instrument_config in config.instruments.items():
        selected_model = args.audio_to_midi_model or instrument_config.model
        selected_checkpoint = args.audio_to_midi_checkpoint
        if selected_checkpoint is None:
            selected_checkpoint = (
                "large"
                if args.audio_to_midi_model == "muscriptor"
                else "2.0"
                if args.audio_to_midi_model == "transkun"
                else instrument_config.checkpoint
            )
        resolved[instrument] = replace(
            instrument_config,
            model=selected_model,
            checkpoint=selected_checkpoint,
            device=args.device or instrument_config.device,
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
    return resolved


def expected_output_files(
    directory: Path,
    output_name: str,
    stem_instruments: tuple[str, ...],
    midi_instruments: tuple[str, ...],
    beats_enabled: bool,
) -> tuple[Path, ...]:
    files = [directory / f"{output_name}.mp3"]
    if beats_enabled:
        files.append(directory / f"{output_name}_beat.json")
    for instrument in stem_instruments:
        files.append(directory / instrument / f"{output_name}_{instrument}.mp3")
    for instrument in midi_instruments:
        files.append(directory / instrument / f"{output_name}_{instrument}.mid")
    return tuple(files)


def output_is_complete(
    output_root: Path,
    output_name: str,
    stem_instruments: tuple[str, ...] | None,
    midi_instruments: tuple[str, ...],
    beats_enabled: bool,
) -> bool:
    if stem_instruments is None:
        return False
    destination = output_root / output_name
    return all(
        path.is_file()
        for path in expected_output_files(
            destination,
            output_name,
            stem_instruments,
            midi_instruments,
            beats_enabled,
        )
    )


def clear_cuda_cache() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_separator(separator: object) -> None:
    model_instance = getattr(separator, "model_instance", None)
    if hasattr(separator, "model_instance"):
        separator.model_instance = None
    del model_instance
    clear_cuda_cache()


def release_audio_to_midi_step(step: AudioToMidiStep) -> None:
    model = getattr(step.model, "model", None)
    step.release()
    del model
    clear_cuda_cache()


def release_beat_detector(detector: BeatDetector) -> None:
    model = getattr(detector, "detector", None)
    detector.release()
    del model
    clear_cuda_cache()


def generate_beats(
    source: Path,
    destination: Path,
    config: BeatDetectionConfig,
    audio_reference: str,
) -> None:
    if not config.enabled:
        return
    detector = BeatDetector(config.model, config.device, config.dbn)
    try:
        write_result(detector, source, destination, audio_reference)
    finally:
        release_beat_detector(detector)


def process_instrument(
    instrument: str,
    stem_path: Path,
    result_dir: Path,
    work_dir: Path,
    output_name: str,
    config: InstrumentAudioToMidiConfig,
) -> None:
    instrument_dir = result_dir / instrument
    instrument_dir.mkdir(parents=True)
    gated_path = work_dir / "gated" / f"{instrument}.wav"
    audio_path = instrument_dir / f"{output_name}_{instrument}.mp3"
    midi_path = instrument_dir / f"{output_name}_{instrument}.mid"
    gate_file(stem_path, gated_path)
    convert_to_mp3(gated_path, audio_path)
    print(f"Gated stem: {audio_path}")

    step = AudioToMidiStep(config)
    try:
        step.run(gated_path, midi_path)
    finally:
        release_audio_to_midi_step(step)
    key, bpm, delay, _, _, _ = normalize_file(midi_path, midi_path)
    print(
        f"Normalized MIDI: {midi_path} "
        f"(key={key}, bpm={bpm:.2f}, delay={delay * 1000:.1f}ms)"
    )
    quantize_midi(midi_path, midi_path, gated_path)
    print(f"Quantized MIDI: {midi_path}")


def write_stem_audio(
    instrument: str,
    stem_path: Path,
    result_dir: Path,
    work_dir: Path,
    output_name: str,
    config: InstrumentAudioToMidiConfig | None,
) -> None:
    instrument_dir = result_dir / instrument
    if config is None:
        instrument_dir.mkdir(parents=True)
        audio_path = instrument_dir / f"{output_name}_{instrument}.mp3"
        convert_to_mp3(stem_path, audio_path)
        print(f"Stored separated stem without transcription: {audio_path}")
        return
    process_instrument(
        instrument,
        stem_path,
        result_dir,
        work_dir,
        output_name,
        config,
    )


def process_audio(
    source: Path,
    output_name: str,
    output_dir: Path,
    work_dir: Path,
    separation_model: str,
    instrument_configs: dict[str, InstrumentAudioToMidiConfig],
    beat_config: BeatDetectionConfig,
) -> None:
    destination_dir = output_dir / output_name
    stem_dir = work_dir / "stems"
    result_dir = work_dir / "result"
    result_dir.mkdir(parents=True)
    original_name = f"{output_name}.mp3"
    convert_to_mp3(source, result_dir / original_name)
    print(f"Stored original audio: {result_dir / original_name}")

    generate_beats(
        source,
        result_dir / f"{output_name}_beat.json",
        beat_config,
        original_name,
    )

    separator = create_separator(stem_dir, separation_model)
    try:
        output_files = separate_with_loaded_model(separator, source, stem_dir)
        print(f"Generated {len(output_files)} stem file(s): {', '.join(output_files)}")
    finally:
        release_separator(separator)

    stems = discover_instrument_stems(stem_dir)
    missing_configured_stems = sorted(set(instrument_configs) - set(stems))
    if missing_configured_stems:
        raise RuntimeError(
            "Configured instruments were not produced by separation: "
            + ", ".join(missing_configured_stems)
        )

    for instrument, stem_path in stems.items():
        instrument_config = instrument_configs.get(instrument)
        if instrument_config is not None:
            print(f"Processing configured instrument: {instrument}")
        write_stem_audio(
            instrument,
            stem_path,
            result_dir,
            work_dir,
            output_name,
            instrument_config,
        )

    missing = [
        path
        for path in expected_output_files(
            result_dir,
            output_name,
            tuple(stems),
            tuple(instrument_configs),
            beat_config.enabled,
        )
        if not path.is_file()
    ]
    if missing:
        names = ", ".join(str(path.relative_to(result_dir)) for path in missing)
        raise RuntimeError(f"Final output is incomplete: {names}")

    if destination_dir.exists():
        if not destination_dir.is_dir():
            raise RuntimeError(f"Output path is not a directory: {destination_dir}")
        shutil.rmtree(destination_dir)
    result_dir.replace(destination_dir)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    instrument_configs = resolve_instrument_configs(config.audio_to_midi, args)

    separation_model = args.model or config.separation_model
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
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
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
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
            failures.append((source, f"Duplicate title/album output name: {output_name}"))
        else:
            unique_audio_files.append((source, output_name))

    stem_instruments = expected_model_stems(separation_model)
    midi_instruments = tuple(instrument_configs)
    processed_count = 0
    skipped_count = 0
    with TemporaryDirectory(prefix="muvisual-batch-", dir=TEMP_DIR) as temporary_dir:
        batch_work_dir = Path(temporary_dir)
        for index, (source, output_name) in enumerate(unique_audio_files, start=1):
            destination_dir = output_dir / output_name
            if output_is_complete(
                output_dir,
                output_name,
                stem_instruments,
                midi_instruments,
                config.beat_detection.enabled,
            ):
                skipped_count += 1
                print(
                    f"\n[Skip {index}/{len(unique_audio_files)}] Already completed: "
                    f"{destination_dir}"
                )
                continue

            print(f"\n[Process {index}/{len(unique_audio_files)}] Processing: {source}")
            try:
                process_audio(
                    source,
                    output_name,
                    output_dir,
                    batch_work_dir / str(index),
                    separation_model,
                    instrument_configs,
                    config.beat_detection,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append((source, str(exc)))
                print(f"Failed: {source}: {exc}", file=sys.stderr)
            else:
                processed_count += 1
                print(f"Completed: {destination_dir}")

    if failures:
        details = "\n".join(f"  {source}: {error}" for source, error in failures)
        raise SystemExit(f"\n{len(failures)} of {len(audio_files)} file(s) failed:\n{details}")
    print(
        f"\nProcessed {processed_count} file(s); "
        f"skipped {skipped_count} completed file(s)."
    )


if __name__ == "__main__":
    main()

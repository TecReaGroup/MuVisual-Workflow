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

from muvisual_workflow.audio_to_midi import AudioToMidiStep
from muvisual_workflow.beat_detection import BeatDetector, MadmomBeatDetector
from muvisual_workflow.core.audio_conversion import convert_to_mp3, convert_to_wav
from muvisual_workflow.separation import (
    AUDIO_EXTENSIONS,
    apply_noise_gate,
    create_separator,
    separate_with_loaded_model,
)
from muvisual_workflow.core.config import (
    AudioToMidiConfig,
    BeatDetectionConfig,
    DEFAULT_CONFIG_PATH,
    WORKFLOW_INSTRUMENT_DIR,
    InstrumentAudioToMidiConfig,
    MuVisualConfig,
    load_config,
)
from muvisual_workflow.core.logging import configure_logging, get_logger
from muvisual_workflow.core.paths import DATA_DIR, PROJECT_ROOT
from muvisual_workflow.midi_quantization import quantize_midi
from muvisual_workflow.music_metadata import (
    analyze_file,
    recognize_chords,
    update_song_metadata,
)

DEFAULT_INPUT = DATA_DIR / "input"
DEFAULT_OUTPUT = DATA_DIR / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
logger = get_logger("pipeline")
INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
STEM_LABEL = re.compile(r"\(([^)]+)\)(?=[_\s.-]|$)", re.IGNORECASE)
BS_ROFORMER_SW_STEMS = ("bass", "drums", "guitar", "other", "piano", "vocals")
CONFIGURED_INSTRUMENT_STEM_NAMES = {"drum": "drums"}


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
        help="Workflow YAML file (default: config/workflow.yaml plus its ordered instruments)",
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
    metadata_enabled: bool,
) -> tuple[Path, ...]:
    files = [directory / f"{output_name}.mp3"]
    if metadata_enabled:
        files.append(directory / f"{output_name}_meta.json")
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
    metadata_enabled: bool,
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
            metadata_enabled,
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


def release_beat_detector(detector: BeatDetector | MadmomBeatDetector) -> None:
    model = getattr(detector, "detector", None)
    detector.release()
    del model
    clear_cuda_cache()


def generate_beats(
    source: Path,
    destination: Path,
    config: BeatDetectionConfig,
    audio_reference: str,
) -> list[float]:
    if not config.enabled:
        return []

    if config.algorithm == "madmom":
        detector = MadmomBeatDetector(config.minimum_bpm, config.maximum_bpm)
        try:
            result = detector.detect_result(source)
            update_song_metadata(
                destination,
                audio=audio_reference,
                beats=result.beats,
                downbeats=result.downbeats,
                beat_metadata={
                    "algorithm": "madmom",
                    "model": result.model,
                    "bpm": result.bpm,
                    "raw_bpm": result.raw_bpm,
                    "doubled_bpm": result.doubled_bpm,
                },
            )
            return result.beats
        finally:
            release_beat_detector(detector)

    detector = BeatDetector(config.model, config.device, config.dbn)
    try:
        detected_beats, downbeats = detector.detect(source)
        beats = [float(value) for value in detected_beats]
        update_song_metadata(
            destination,
            audio=audio_reference,
            beats=beats,
            downbeats=[float(value) for value in downbeats],
            beat_metadata={
                "algorithm": "beat_this",
                "model": config.model,
            },
        )
        return beats
    finally:
        release_beat_detector(detector)

def store_stem_audio(
    instrument: str,
    stem_path: Path,
    result_dir: Path,
    output_name: str,
) -> None:
    instrument_dir = result_dir / instrument
    instrument_dir.mkdir(parents=True, exist_ok=True)
    audio_path = instrument_dir / f"{output_name}_{instrument}.mp3"
    convert_to_mp3(stem_path, audio_path)
    logger.info("Stored separated stem: %s", audio_path)



def restore_configured_stems(
    result_dir: Path,
    stem_dir: Path,
    output_name: str,
    instruments: tuple[str, ...],
) -> dict[str, Path]:
    """Decode separated audio saved by the main workflow for transcription."""
    restored_stems: dict[str, Path] = {}
    for instrument in instruments:
        stem_name = CONFIGURED_INSTRUMENT_STEM_NAMES.get(instrument, instrument)
        stored_stem = result_dir / stem_name / f"{output_name}_{stem_name}.mp3"
        if not stored_stem.is_file():
            continue
        restored_stem = stem_dir / f"{instrument}.wav"
        convert_to_wav(stored_stem, restored_stem)
        restored_stems[instrument] = restored_stem
        logger.info("Restored separated stem for transcription: %s", restored_stem)
    return restored_stems

def transcribe_instrument(
    instrument: str,
    stem_path: Path,
    result_dir: Path,
    output_name: str,
    config: InstrumentAudioToMidiConfig,
) -> Path:
    midi_path = result_dir / instrument / f"{output_name}_{instrument}.mid"
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    step = AudioToMidiStep(config)
    try:
        step.run(stem_path, midi_path)
    finally:
        release_audio_to_midi_step(step)
    logger.info("Transcribed MIDI: %s", midi_path)
    return midi_path


def process_audio(
    source: Path,
    output_name: str,
    output_dir: Path,
    work_dir: Path,
    config: MuVisualConfig,
) -> None:
    if not config.enabled:
        raise RuntimeError(
            f"Instrument workflow {config.instrument!r} is disabled"
        )
    destination_dir = output_dir / output_name
    raw_stem_dir = work_dir / "raw_stems"
    stem_dir = work_dir / "stems"
    result_dir = work_dir / "result"
    result_dir.mkdir(parents=True)
    if destination_dir.exists():
        if not destination_dir.is_dir():
            raise RuntimeError(f"Output path is not a directory: {destination_dir}")
        shutil.copytree(destination_dir, result_dir, dirs_exist_ok=True)
    original_name = f"{output_name}.mp3"
    (result_dir / f"{output_name}_beat.json").unlink(missing_ok=True)
    convert_to_mp3(source, result_dir / original_name)
    logger.info("Stored original audio: %s", result_dir / original_name)

    instrument_configs = (
        config.audio_to_midi.instruments if config.audio_to_midi is not None else {}
    )
    stems = restore_configured_stems(
        result_dir,
        stem_dir,
        output_name,
        tuple(instrument_configs),
    )
    midi_files: dict[str, Path] = {}
    metadata_path = result_dir / f"{output_name}_meta.json"
    metadata_enabled = False
    detected_beats: list[float] = []

    for workflow_step in config.workflow:
        logger.info(
            "Running workflow step %s with option %s",
            workflow_step.name,
            workflow_step.option,
        )
        if workflow_step.name == "beat_detection":
            beat_config = config.require_beat_detection()
            metadata_enabled = beat_config.enabled
            detected_beats = generate_beats(
                source,
                metadata_path,
                beat_config,
                original_name,
            )
            continue

        if workflow_step.name == "separation":
            separation_config = config.require_separation()
            if not separation_config.enabled:
                logger.info("Workflow step separation is disabled")
                continue
            separator = create_separator(raw_stem_dir, separation_config.model)
            try:
                output_files = separate_with_loaded_model(
                    separator, source, raw_stem_dir
                )
                logger.info(
                    "Generated %d stem file(s): %s",
                    len(output_files),
                    ", ".join(output_files),
                )
            finally:
                release_separator(separator)

            raw_stems = discover_instrument_stems(raw_stem_dir)
            stems = {}
            for instrument, raw_stem_path in raw_stems.items():
                stem_path = stem_dir / f"{instrument}.wav"
                apply_noise_gate(
                    raw_stem_path,
                    stem_path,
                    separation_config.noise_gate,
                )
                stems[instrument] = stem_path
                store_stem_audio(instrument, stem_path, result_dir, output_name)
            if separation_config.noise_gate.enabled:
                logger.info("Applied BS-Roformer noise gate to %d stem(s)", len(stems))
            continue

        if workflow_step.name == "audio_to_midi":
            audio_to_midi_config = config.require_audio_to_midi()
            if not audio_to_midi_config.enabled:
                logger.info("Workflow step audio_to_midi is disabled")
                continue
            missing_configured_stems = sorted(
                set(audio_to_midi_config.instruments) - set(stems)
            )
            if missing_configured_stems and stems:
                raise RuntimeError(
                    "Configured instruments were not produced by separation: "
                    + ", ".join(missing_configured_stems)
                )
            for instrument, instrument_config in audio_to_midi_config.instruments.items():
                logger.info("Transcribing configured instrument: %s", instrument)
                midi_files[instrument] = transcribe_instrument(
                    instrument,
                    stems.get(instrument, source),
                    result_dir,
                    output_name,
                    instrument_config,
                )
            continue

        if workflow_step.name == "music_metadata":
            metadata_config = config.require_music_metadata(workflow_step.option)
            metadata_enabled = metadata_enabled or metadata_config.enabled
            if not metadata_config.enabled:
                continue

            if metadata_config.chord_recognition is not None:
                if not detected_beats:
                    raise RuntimeError(
                        "music_metadata chord recognition requires enabled "
                        "beat_detection output"
                    )
                chord_payload = recognize_chords(
                    source,
                    detected_beats,
                    metadata_config.chord_recognition,
                )
                update_song_metadata(
                    metadata_path,
                    audio=original_name,
                    chords=chord_payload,
                )
                logger.info(
                    "Recognized %d chord segment(s) with Chord-CNN-LSTM",
                    len(chord_payload["segments"]),
                )

            if metadata_config.key_bpm_delay is not None:
                if not midi_files:
                    raise RuntimeError(
                        "music_metadata key_bpm_delay requires audio_to_midi output"
                    )
                for instrument, midi_path in midi_files.items():
                    metadata = analyze_file(
                        midi_path,
                        metadata_config.key_bpm_delay.alignment_sample_count,
                    )
                    update_song_metadata(
                        metadata_path,
                        audio=original_name,
                        instrument=instrument,
                        instrument_metadata=metadata,
                    )
                    logger.info(
                        "Recognized music metadata for %s: "
                        "key=%s, bpm=%.2f, delay=%.1f ms",
                        instrument,
                        metadata.key,
                        metadata.bpm,
                        metadata.delay * 1000,
                    )
            continue
        if workflow_step.name == "midi_quantization":
            quantization_config = config.require_midi_quantization()
            if not quantization_config.enabled:
                logger.info("Workflow step midi_quantization is disabled")
                continue
            if not midi_files:
                raise RuntimeError("midi_quantization requires audio_to_midi output")
            for instrument, midi_path in midi_files.items():
                quantize_midi(
                    midi_path,
                    midi_path,
                    stems.get(instrument, source),
                    quantization_config,
                )
                logger.info("Quantized MIDI: %s", midi_path)
            continue

        raise RuntimeError(f"Unsupported workflow step: {workflow_step.name}")

    stored_stem_instruments = (
        tuple(stems)
        if config.separation is not None and config.separation.enabled
        else ()
    )
    missing = [
        path
        for path in expected_output_files(
            result_dir,
            output_name,
            stored_stem_instruments,
            tuple(instrument_configs),
            metadata_enabled,
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


def resolve_runtime_config(
    config: MuVisualConfig,
    args: argparse.Namespace,
) -> MuVisualConfig:
    separation = config.separation
    if args.model is not None:
        if separation is None:
            raise ValueError("--separation-model requires a configured separation step")
        separation = replace(separation, model=args.model)

    audio_to_midi = config.audio_to_midi
    audio_overridden = any(
        value is not None
        for value in (
            args.audio_to_midi_model,
            args.audio_to_midi_checkpoint,
            args.device,
            args.segment_hop_size,
            args.segment_size,
        )
    )
    if audio_overridden and audio_to_midi is None:
        raise ValueError("Audio-to-MIDI overrides require a configured audio_to_midi step")
    if audio_to_midi is not None:
        audio_to_midi = replace(
            audio_to_midi,
            instruments=resolve_instrument_configs(audio_to_midi, args),
        )
    return replace(config, separation=separation, audio_to_midi=audio_to_midi)


def run_workflow(config: MuVisualConfig, args: argparse.Namespace) -> None:
    config = resolve_runtime_config(config, args)
    if not config.enabled:
        logger.info("Instrument workflow %s is disabled", config.instrument)
        return
    logger.info("Selected instrument workflow: %s", config.instrument)
    instrument_configs = (
        config.audio_to_midi.instruments if config.audio_to_midi is not None else {}
    )
    separation_model = config.separation.model if config.separation is not None else None
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

    separation_enabled = config.separation is not None and config.separation.enabled
    audio_to_midi_enabled = config.audio_to_midi is not None and config.audio_to_midi.enabled
    stem_instruments = (
        expected_model_stems(separation_model)
        if separation_enabled and separation_model is not None
        else ()
    )
    midi_instruments = tuple(instrument_configs) if audio_to_midi_enabled else ()
    metadata_enabled = any(
        metadata_config.enabled for metadata_config in config.music_metadata
    ) or (
        config.beat_detection is not None and config.beat_detection.enabled
    )
    processed_count = 0
    skipped_count = 0
    with TemporaryDirectory(prefix="muvisual-batch-", dir=TEMP_DIR) as temporary_dir:
        batch_work_dir = Path(temporary_dir)
        for index, (source, output_name) in enumerate(unique_audio_files, start=1):
            destination_dir = output_dir / output_name
            if not config.overwrite and output_is_complete(
                output_dir,
                output_name,
                stem_instruments,
                midi_instruments,
                metadata_enabled,
            ):
                skipped_count += 1
                logger.info(
                    "[Skip %d/%d] Already completed: %s",
                    index,
                    len(unique_audio_files),
                    destination_dir,
                )
                continue

            logger.info(
                "[Process %d/%d] Processing: %s",
                index,
                len(unique_audio_files),
                source,
            )
            try:
                process_audio(
                    source,
                    output_name,
                    output_dir,
                    batch_work_dir / str(index),
                    config,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append((source, str(exc)))
                logger.error("Failed: %s: %s", source, exc)
            else:
                processed_count += 1
                logger.info("Completed: %s", destination_dir)

    if failures:
        details = "\n".join(f"  {source}: {error}" for source, error in failures)
        logger.error("%d of %d file(s) failed:\n%s", len(failures), len(audio_files), details)
        raise SystemExit(f"\n{len(failures)} of {len(audio_files)} file(s) failed:\n{details}")
    logger.info(
        "Processed %d file(s); skipped %d completed file(s).",
        processed_count,
        skipped_count,
    )



def main() -> None:
    configure_logging()
    args = parse_args()
    if args.config is not None:
        configs = [load_config(args.config.expanduser().resolve())]
    else:
        main_config = load_config(DEFAULT_CONFIG_PATH)
        configs = [main_config]
        for instrument in main_config.instrument_order:
            instrument_path = WORKFLOW_INSTRUMENT_DIR / f"workflow_{instrument}.yaml"
            instrument_config = load_config(instrument_path)
            if instrument_config.instrument != instrument:
                raise ValueError(
                    f"Instrument workflow {instrument_path} configures "
                    f"{instrument_config.instrument!r}, expected {instrument!r}"
                )
            configs.append(instrument_config)

    enabled_configs = [config for config in configs if config.enabled]
    if not enabled_configs:
        logger.info("No enabled workflows found")
        return

    for config in enabled_configs:
        run_workflow(config, args)
if __name__ == "__main__":
    main()

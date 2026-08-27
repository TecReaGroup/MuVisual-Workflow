"""Load and validate MuVisual workflow and step-option YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, cast

import yaml

from muvisual_workflow.core.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "workflow.yaml"
SUPPORTED_STEPS = (
    "beat_detection",
    "separation",
    "audio_to_midi",
    "music_metadata",
    "midi_quantization",
)
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    option: str


@dataclass(frozen=True)
class NoiseGateConfig:
    enabled: bool = True
    threshold_db: float = -48.0
    attack_ms: float = 8.0
    hold_ms: float = 80.0
    release_ms: float = 180.0
    analysis_ms: float = 5.0


@dataclass(frozen=True)
class SeparationConfig:
    enabled: bool
    model: str
    noise_gate: NoiseGateConfig


@dataclass(frozen=True)
class InstrumentAudioToMidiConfig:
    instrument: str
    model: str
    checkpoint: str
    device: str
    dtype: str | None
    target_instruments: tuple[str, ...]
    segment_hop_size: int | None = None
    segment_size: int | None = None


@dataclass(frozen=True)
class AudioToMidiConfig:
    enabled: bool
    default_instrument: str
    instruments: dict[str, InstrumentAudioToMidiConfig]

    def for_instrument(self, instrument: str | None = None) -> InstrumentAudioToMidiConfig:
        name = (instrument or self.default_instrument).casefold()
        try:
            return self.instruments[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.instruments))
            raise ValueError(
                f"No audio-to-MIDI configuration for instrument {name!r}; "
                f"available: {available}"
            ) from exc


@dataclass(frozen=True)
class KeyBpmDelayConfig:
    enabled: bool
    algorithm: str
    alignment_sample_count: int = 35


@dataclass(frozen=True)
class ChordRecognitionConfig:
    enabled: bool
    algorithm: str
    repository_path: Path
    chord_dictionary: str


@dataclass(frozen=True)
class MusicMetadataConfig:
    option: str
    enabled: bool
    key_bpm_delay: KeyBpmDelayConfig | None = None
    chord_recognition: ChordRecognitionConfig | None = None


@dataclass(frozen=True)
class MidiQuantizationConfig:
    enabled: bool = True
    hand_split_note: int = 60
    simultaneous_threshold_ticks: int = 100
    minimum_note_ticks: int = 100
    next_group_gap_ticks: int = 10


@dataclass(frozen=True)
class BeatDetectionConfig:
    enabled: bool
    algorithm: str
    model: str
    device: str
    dbn: bool
    minimum_bpm: float = 90.0
    maximum_bpm: float = 180.0


@dataclass(frozen=True)
class MuVisualConfig:
    enabled: bool
    overwrite: bool
    instrument: str
    workflow: tuple[WorkflowStep, ...]
    separation: SeparationConfig | None = None
    audio_to_midi: AudioToMidiConfig | None = None
    music_metadata: tuple[MusicMetadataConfig, ...] = ()
    midi_quantization: MidiQuantizationConfig | None = None
    beat_detection: BeatDetectionConfig | None = None

    def has_step(self, name: str) -> bool:
        return any(step.name == name for step in self.workflow)

    def require_separation(self) -> SeparationConfig:
        if self.separation is None:
            raise ValueError("The workflow does not configure the separation step")
        return self.separation

    def require_audio_to_midi(self) -> AudioToMidiConfig:
        if self.audio_to_midi is None:
            raise ValueError("The workflow does not configure the audio_to_midi step")
        return self.audio_to_midi

    def require_music_metadata(self, option: str) -> MusicMetadataConfig:
        for metadata_config in self.music_metadata:
            if metadata_config.option == option:
                return metadata_config
        raise ValueError(
            f"The workflow does not configure music_metadata option {option!r}"
        )

    def require_midi_quantization(self) -> MidiQuantizationConfig:
        if self.midi_quantization is None:
            raise ValueError("The workflow does not configure the midi_quantization step")
        return self.midi_quantization

    def require_beat_detection(self) -> BeatDetectionConfig:
        if self.beat_detection is None:
            raise ValueError("The workflow does not configure the beat_detection step")
        return self.beat_detection


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    return cast(dict[str, Any], value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _name(value: object, field_name: str) -> str:
    name = _string(value, field_name)
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{field_name} may only contain letters, numbers, underscores, and hyphens"
        )
    return name


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false")
    return value


def _number(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _non_negative_number(value: object, field_name: str) -> float:
    number = _number(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _positive_number(value: object, field_name: str) -> float:
    number = _number(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be positive")
    return number


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _midi_note(value: object, field_name: str) -> int:
    note = _non_negative_int(value, field_name)
    if note > 127:
        raise ValueError(f"{field_name} must be between 0 and 127")
    return note


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None or value == "auto":
        return None
    return _string(value, field_name)


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _device(value: object, field_name: str) -> str:
    device = _string(value, field_name)
    if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
        raise ValueError(f"{field_name} must be auto, cpu, cuda, or cuda:N")
    return device


def _target_instruments(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(instrument, str) and instrument.strip() for instrument in value
    ):
        raise ValueError(f"{field_name} must be a non-empty list of strings")
    return tuple(instrument.strip() for instrument in value)


def _load_yaml(path: Path, field_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if payload is None:
        raise ValueError(f"{field_name} is empty: {path}")
    return _mapping(payload, field_name)


def _load_workflow(payload: dict[str, Any]) -> tuple[WorkflowStep, ...]:
    raw_workflow = payload.get("workflow")
    if not isinstance(raw_workflow, list) or not raw_workflow:
        raise ValueError("workflow must be a non-empty list")

    workflow: list[WorkflowStep] = []
    seen: set[str] = set()
    for index, raw_step in enumerate(raw_workflow):
        entry = _mapping(raw_step, f"workflow[{index}]")
        step = _name(entry.get("step"), f"workflow[{index}].step").casefold()
        option = _name(entry.get("option"), f"workflow[{index}].option")
        if step not in SUPPORTED_STEPS:
            supported = ", ".join(SUPPORTED_STEPS)
            raise ValueError(f"Unsupported workflow step {step!r}; supported: {supported}")
        if step in seen and step != "music_metadata":
            raise ValueError(f"Workflow step {step!r} is configured more than once")
        unexpected = set(entry) - {"step", "option"}
        if unexpected:
            raise ValueError(
                f"workflow[{index}] has unsupported fields: {', '.join(sorted(unexpected))}"
            )
        workflow.append(WorkflowStep(step, option))
        seen.add(step)

    names = [step.name for step in workflow]
    dependencies: list[tuple[str, str]] = []
    for workflow_step in workflow:
        if workflow_step.name == "music_metadata":
            if workflow_step.option == "key_bpm_delay":
                dependencies.append(("music_metadata:key_bpm_delay", "audio_to_midi"))
            elif workflow_step.option == "chord_recognition":
                dependencies.append(("music_metadata:chord_recognition", "beat_detection"))
        elif workflow_step.name == "midi_quantization":
            dependencies.append(("midi_quantization", "audio_to_midi"))

    for step, dependency in dependencies:
        step_name = step.split(":", maxsplit=1)[0]
        if dependency not in seen:
            raise ValueError(f"{step} requires the {dependency} step")
        if names.index(step_name) < names.index(dependency):
            raise ValueError(f"{dependency} must appear before {step}")
    return tuple(workflow)


def _instrument_config(
    name: str,
    payload: dict[str, Any],
    defaults: dict[str, object],
    source: str,
) -> InstrumentAudioToMidiConfig:
    prefix = f"{source}.instruments.{name}"
    model = _string(payload.get("model", "muscriptor"), f"{prefix}.model").casefold()
    if model not in {"muscriptor", "transkun"}:
        raise ValueError(f"{prefix}.model must be 'muscriptor' or 'transkun'")

    default_checkpoint = "large" if model == "muscriptor" else "2.0"
    checkpoint = _string(payload.get("checkpoint", default_checkpoint), f"{prefix}.checkpoint")
    if model == "muscriptor" and checkpoint not in {"small", "medium", "large"}:
        raise ValueError(f"{prefix}.checkpoint must be small, medium, or large")

    return InstrumentAudioToMidiConfig(
        instrument=name,
        model=model,
        checkpoint=checkpoint,
        device=_device(payload.get("device", defaults["device"]), f"{prefix}.device"),
        dtype=_optional_string(payload.get("dtype", defaults["dtype"]), f"{prefix}.dtype"),
        target_instruments=_target_instruments(
            payload.get("target_instruments", [name]),
            f"{prefix}.target_instruments",
        ),
        segment_hop_size=_optional_positive_int(
            payload.get("segment_hop_size", defaults["segment_hop_size"]),
            f"{prefix}.segment_hop_size",
        ),
        segment_size=_optional_positive_int(
            payload.get("segment_size", defaults["segment_size"]),
            f"{prefix}.segment_size",
        ),
    )


def _parse_audio_to_midi(source: str, payload: dict[str, Any]) -> AudioToMidiConfig:
    default_instrument = _string(
        payload.get("default_instrument", "piano"),
        f"{source}.default_instrument",
    ).casefold()
    instrument_tables = _mapping(payload.get("instruments", {}), f"{source}.instruments")
    defaults: dict[str, object] = {
        "device": _device(payload.get("device", "auto"), f"{source}.device"),
        "dtype": payload.get("dtype"),
        "segment_hop_size": payload.get("segment_hop_size"),
        "segment_size": payload.get("segment_size"),
    }
    instruments: dict[str, InstrumentAudioToMidiConfig] = {}
    for raw_name, value in instrument_tables.items():
        name = _name(raw_name, f"{source}.instruments key").casefold()
        instruments[name] = _instrument_config(
            name,
            _mapping(value, f"{source}.instruments.{raw_name}"),
            defaults,
            source,
        )
    if not instruments:
        raise ValueError(f"{source}.instruments must configure at least one instrument")
    if default_instrument not in instruments:
        available = ", ".join(sorted(instruments))
        raise ValueError(
            f"{source}.default_instrument {default_instrument!r} is not configured; "
            f"available: {available}"
        )
    return AudioToMidiConfig(
        enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
        default_instrument=default_instrument,
        instruments=instruments,
    )


def _parse_step_option(step: WorkflowStep, payload: dict[str, Any]) -> object:
    source = f"{step.name}/{step.option}.yaml"
    if step.name == "separation":
        noise_gate_payload = _mapping(payload.get("noise_gate", {}), f"{source}.noise_gate")
        return SeparationConfig(
            enabled=_boolean(payload.get('enabled', True), f'{source}.enabled'),
            model=_string(payload.get("model", "BS-Roformer-SW.ckpt"), f"{source}.model"),
            noise_gate=NoiseGateConfig(
                enabled=_boolean(noise_gate_payload.get("enabled", True), f"{source}.noise_gate.enabled"),
                threshold_db=_number(noise_gate_payload.get("threshold_db", -48.0), f"{source}.noise_gate.threshold_db"),
                attack_ms=_non_negative_number(noise_gate_payload.get("attack_ms", 8.0), f"{source}.noise_gate.attack_ms"),
                hold_ms=_non_negative_number(noise_gate_payload.get("hold_ms", 80.0), f"{source}.noise_gate.hold_ms"),
                release_ms=_non_negative_number(noise_gate_payload.get("release_ms", 180.0), f"{source}.noise_gate.release_ms"),
                analysis_ms=_positive_number(noise_gate_payload.get("analysis_ms", 5.0), f"{source}.noise_gate.analysis_ms"),
            ),
        )
    if step.name == "beat_detection":
        algorithm = _name(payload.get("algorithm", "beat_this"), f"{source}.algorithm").casefold()
        if algorithm not in {"beat_this", "madmom"}:
            raise ValueError(f"{source}.algorithm must be beat_this or madmom")
        minimum_bpm = _positive_number(payload.get("minimum_bpm", 90.0), f"{source}.minimum_bpm")
        maximum_bpm = _positive_number(payload.get("maximum_bpm", 180.0), f"{source}.maximum_bpm")
        if maximum_bpm <= minimum_bpm:
            raise ValueError(f"{source}.maximum_bpm must be greater than minimum_bpm")
        return BeatDetectionConfig(
            enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
            algorithm=algorithm,
            model=_string(payload.get("model", "final0"), f"{source}.model"),
            device=_device(payload.get("device", "auto"), f"{source}.device"),
            dbn=_boolean(payload.get("dbn", False), f"{source}.dbn"),
            minimum_bpm=minimum_bpm,
            maximum_bpm=maximum_bpm,
        )
    if step.name == "audio_to_midi":
        return _parse_audio_to_midi(source, payload)
    if step.name == "music_metadata":
        if step.option == "key_bpm_delay":
            return MusicMetadataConfig(
                option=step.option,
                enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
                key_bpm_delay=KeyBpmDelayConfig(
                    enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
                    algorithm="key_bpm_delay",
                    alignment_sample_count=_positive_int(
                        payload.get("alignment_sample_count", 35),
                        f"{source}.alignment_sample_count",
                    ),
                ),
            )
        if step.option == "chord_recognition":
            algorithm = _name(
                payload.get("algorithm", "chord_cnn_lstm"),
                f"{source}.algorithm",
            ).casefold()
            if algorithm != "chord_cnn_lstm":
                raise ValueError(f"{source}.algorithm must be chord_cnn_lstm")
            chord_dictionary = _name(
                payload.get("chord_dictionary", "submission"),
                f"{source}.chord_dictionary",
            ).casefold()
            supported = {"submission", "ismir2017", "full", "extended"}
            if chord_dictionary not in supported:
                raise ValueError(
                    f"{source}.chord_dictionary must be one of: "
                    + ", ".join(sorted(supported))
                )
            return MusicMetadataConfig(
                option=step.option,
                enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
                chord_recognition=ChordRecognitionConfig(
                    enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
                    algorithm=algorithm,
                    repository_path=Path(
                        _string(
                            payload.get("repository_path", "data/model/chord_cnn_lstm"),
                            f"{source}.repository_path",
                        )
                    ),
                    chord_dictionary=chord_dictionary,
                ),
            )
        raise ValueError(
            f"Unsupported music_metadata option: {step.option}; "
            "expected key_bpm_delay or chord_recognition"
        )
    if step.name == "midi_quantization":
        return MidiQuantizationConfig(
            enabled=_boolean(payload.get('enabled', True), f'{source}.enabled'),
            hand_split_note=_midi_note(payload.get("hand_split_note", 60), f"{source}.hand_split_note"),
            simultaneous_threshold_ticks=_non_negative_int(
                payload.get("simultaneous_threshold_ticks", 100),
                f"{source}.simultaneous_threshold_ticks",
            ),
            minimum_note_ticks=_positive_int(
                payload.get("minimum_note_ticks", 100),
                f"{source}.minimum_note_ticks",
            ),
            next_group_gap_ticks=_non_negative_int(
                payload.get("next_group_gap_ticks", 10),
                f"{source}.next_group_gap_ticks",
            ),
        )
    raise ValueError(f"Unsupported workflow step: {step.name}")


def load_config(path: Path | None = None) -> MuVisualConfig:
    """Load a workflow file and each selected config/<step>/<option>.yaml file."""
    workflow_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    workflow_payload = _load_yaml(workflow_path, "Workflow configuration")
    unexpected = set(workflow_payload) - {"enabled", "overwrite", "instrument", "workflow"}
    if unexpected:
        raise ValueError(
            "Workflow configuration has unsupported fields: "
            + ", ".join(sorted(unexpected))
        )
    enabled = _boolean(workflow_payload.get("enabled", True), "enabled")
    overwrite = _boolean(workflow_payload.get("overwrite", False), "overwrite")
    instrument = _name(
        workflow_payload.get("instrument"),
        "instrument",
    ).casefold()
    workflow = _load_workflow(workflow_payload)
    config_root = workflow_path.parent

    loaded: dict[str, object] = {}
    metadata_configs: list[MusicMetadataConfig] = []
    for step in workflow:
        option_path = config_root / step.name / f"{step.option}.yaml"
        payload = _load_yaml(option_path, f"Configuration option for {step.name}")
        parsed = _parse_step_option(step, payload)
        if step.name == "music_metadata":
            metadata_configs.append(cast(MusicMetadataConfig, parsed))
        else:
            loaded[step.name] = parsed

    audio_to_midi = cast(AudioToMidiConfig | None, loaded.get("audio_to_midi"))
    if audio_to_midi is not None:
        configured_instruments = set(audio_to_midi.instruments)
        if configured_instruments != {instrument}:
            configured = ", ".join(sorted(configured_instruments))
            raise ValueError(
                f"Instrument workflow {instrument!r} must configure only that instrument; "
                f"configured: {configured}"
            )

    return MuVisualConfig(
        enabled=enabled,
        overwrite=overwrite,
        instrument=instrument,
        workflow=workflow,
        separation=cast(SeparationConfig | None, loaded.get("separation")),
        audio_to_midi=audio_to_midi,
        music_metadata=tuple(metadata_configs),
        midi_quantization=cast(MidiQuantizationConfig | None, loaded.get("midi_quantization")),
        beat_detection=cast(BeatDetectionConfig | None, loaded.get("beat_detection")),
    )

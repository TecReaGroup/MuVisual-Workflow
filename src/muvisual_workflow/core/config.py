"""Load and validate MuVisual workflow and step-option YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, cast

import yaml

from muvisual_workflow.core.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "workflow.yaml"
SUPPORTED_STEPS = ("beat_detection", "separation", "audio_to_midi")
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    option: str


@dataclass(frozen=True)
class SeparationConfig:
    model: str


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
class BeatDetectionConfig:
    enabled: bool
    model: str
    device: str
    dbn: bool


@dataclass(frozen=True)
class MuVisualConfig:
    workflow: tuple[WorkflowStep, ...]
    separation: SeparationConfig | None = None
    audio_to_midi: AudioToMidiConfig | None = None
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


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None or value == "auto":
        return None
    return _string(value, field_name)


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


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
        if step in seen:
            raise ValueError(f"Workflow step {step!r} is configured more than once")
        unexpected = set(entry) - {"step", "option"}
        if unexpected:
            raise ValueError(
                f"workflow[{index}] has unsupported fields: {', '.join(sorted(unexpected))}"
            )
        workflow.append(WorkflowStep(step, option))
        seen.add(step)

    names = [step.name for step in workflow]
    if "audio_to_midi" in seen:
        if "separation" not in seen:
            raise ValueError("audio_to_midi requires the separation step")
        if names.index("audio_to_midi") < names.index("separation"):
            raise ValueError("separation must appear before audio_to_midi in workflow")
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


def _parse_step_option(step: WorkflowStep, payload: dict[str, Any]) -> object:
    source = f"{step.name}/{step.option}.yaml"
    if step.name == "separation":
        return SeparationConfig(
            model=_string(payload.get("model", "BS-Roformer-SW.ckpt"), f"{source}.model")
        )

    if step.name == "beat_detection":
        return BeatDetectionConfig(
            enabled=_boolean(payload.get("enabled", True), f"{source}.enabled"),
            model=_string(payload.get("model", "final0"), f"{source}.model"),
            device=_device(payload.get("device", "auto"), f"{source}.device"),
            dbn=_boolean(payload.get("dbn", False), f"{source}.dbn"),
        )

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
    return AudioToMidiConfig(default_instrument, instruments)


def load_config(path: Path | None = None) -> MuVisualConfig:
    """Load workflow.yaml and each selected config/<step>/<option>.yaml file."""
    workflow_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    workflow_payload = _load_yaml(workflow_path, "Workflow configuration")
    workflow = _load_workflow(workflow_payload)
    config_root = workflow_path.parent

    loaded: dict[str, object] = {}
    for step in workflow:
        option_path = config_root / step.name / f"{step.option}.yaml"
        payload = _load_yaml(option_path, f"Configuration option for {step.name}")
        loaded[step.name] = _parse_step_option(step, payload)

    return MuVisualConfig(
        workflow=workflow,
        separation=cast(SeparationConfig | None, loaded.get("separation")),
        audio_to_midi=cast(AudioToMidiConfig | None, loaded.get("audio_to_midi")),
        beat_detection=cast(BeatDetectionConfig | None, loaded.get("beat_detection")),
    )

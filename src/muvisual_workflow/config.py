"""Load and validate MuVisual runtime configuration from TOML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from muvisual_workflow.paths import PROJECT_ROOT

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "muvisual.toml"


@dataclass(frozen=True)
class InstrumentAudioToMidiConfig:
    instrument: str
    model: str
    checkpoint: str
    device: str
    dtype: str | None
    target_instruments: tuple[str, ...]
    quantize: bool
    segment_hop_size: int | None = None
    segment_size: int | None = None


def _default_instrument_configs() -> dict[str, InstrumentAudioToMidiConfig]:
    piano = InstrumentAudioToMidiConfig(
        instrument="piano",
        model="muscriptor",
        checkpoint="large",
        device="auto",
        dtype=None,
        target_instruments=("acoustic_piano",),
        quantize=True,
    )
    return {"piano": piano}


@dataclass(frozen=True)
class AudioToMidiConfig:
    default_instrument: str = "piano"
    instruments: dict[str, InstrumentAudioToMidiConfig] = field(
        default_factory=_default_instrument_configs
    )

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
    enabled: bool = True
    model: str = "final0"
    device: str = "auto"
    dbn: bool = False


@dataclass(frozen=True)
class MuVisualConfig:
    separation_model: str = "BS-Roformer-SW.ckpt"
    audio_to_midi: AudioToMidiConfig = field(default_factory=AudioToMidiConfig)
    beat_detection: BeatDetectionConfig = BeatDetectionConfig()


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section [{name}] must be a table")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None or value == "auto":
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string or 'auto'")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be true or false")
    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _device(value: object, field_name: str) -> str:
    value = _string(value, field_name)
    if value not in {"auto", "cpu", "cuda"} and not value.startswith("cuda:"):
        raise ValueError(f"{field_name} must be auto, cpu, cuda, or cuda:N")
    return value


def _target_instruments(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(instrument, str) and instrument for instrument in value
    ):
        raise ValueError(f"{field_name} must be a non-empty list of strings")
    return tuple(value)


def _instrument_config(
    name: str,
    payload: dict[str, object],
    defaults: dict[str, object],
) -> InstrumentAudioToMidiConfig:
    prefix = f"audio_to_midi.instruments.{name}"
    model = _string(payload.get("model", "muscriptor"), f"{prefix}.model").casefold()
    if model not in {"muscriptor", "transkun"}:
        raise ValueError(f"{prefix}.model must be 'muscriptor' or 'transkun'")

    default_checkpoint = "large" if model == "muscriptor" else "2.0"
    checkpoint = _string(
        payload.get("checkpoint", default_checkpoint),
        f"{prefix}.checkpoint",
    )
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
        quantize=_boolean(
            payload.get("quantize", defaults["quantize"]),
            f"{prefix}.quantize",
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


def load_config(path: Path | None = None) -> MuVisualConfig:
    """Load a config file, falling back to built-in defaults when absent."""
    config_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not config_path.is_file():
        return MuVisualConfig()

    with config_path.open("rb") as file:
        payload = tomllib.load(file)

    separation = _section(payload, "separation")
    audio_to_midi = _section(payload, "audio_to_midi")
    beat_detection = _section(payload, "beat_detection")
    instrument_tables = _section(audio_to_midi, "instruments")

    defaults: dict[str, object] = {
        "device": _device(audio_to_midi.get("device", "auto"), "audio_to_midi.device"),
        "dtype": audio_to_midi.get("dtype"),
        "quantize": _boolean(audio_to_midi.get("quantize", True), "audio_to_midi.quantize"),
        "segment_hop_size": audio_to_midi.get("segment_hop_size"),
        "segment_size": audio_to_midi.get("segment_size"),
    }
    default_instrument = _string(
        audio_to_midi.get("default_instrument", "piano"),
        "audio_to_midi.default_instrument",
    ).casefold()
    instruments: dict[str, InstrumentAudioToMidiConfig] = {}
    for raw_name, value in instrument_tables.items():
        if not isinstance(value, dict):
            raise ValueError(f"audio_to_midi.instruments.{raw_name} must be a table")
        name = raw_name.casefold()
        instruments[name] = _instrument_config(name, value, defaults)
    if not instruments:
        instruments = _default_instrument_configs()
    if default_instrument not in instruments:
        available = ", ".join(sorted(instruments))
        raise ValueError(
            f"audio_to_midi.default_instrument {default_instrument!r} is not configured; "
            f"available: {available}"
        )

    separation_model = _string(
        separation.get("model", "BS-Roformer-SW.ckpt"),
        "separation.model",
    )
    return MuVisualConfig(
        separation_model=separation_model,
        audio_to_midi=AudioToMidiConfig(default_instrument, instruments),
        beat_detection=BeatDetectionConfig(
            enabled=_boolean(beat_detection.get("enabled", True), "beat_detection.enabled"),
            model=_string(beat_detection.get("model", "final0"), "beat_detection.model"),
            device=_device(beat_detection.get("device", "auto"), "beat_detection.device"),
            dbn=_boolean(beat_detection.get("dbn", False), "beat_detection.dbn"),
        ),
    )

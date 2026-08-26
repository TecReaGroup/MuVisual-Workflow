"""Beat and downbeat detection with Beat This."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
from math import log2
from pathlib import Path
import shutil
from statistics import median
import subprocess
import tempfile
from typing import Any, Iterable

from muvisual_workflow.core.config import load_config
from muvisual_workflow.core.logging import configure_logging, get_logger
from muvisual_workflow.core.paths import DEVELOP_DATA_DIR, PROJECT_ROOT

_OCTAVE_CORRECTION = True
_OCTAVE_TOLERANCE = 0.35
_OCTAVE_MIN_SEGMENT_BEATS = 4
_OCTAVE_FACTORS = (0.5, 1.0, 2.0)


def _normalize_beat_octaves(
    beats: Iterable[float],
    downbeats: Iterable[float],
    tolerance: float = _OCTAVE_TOLERANCE,
    min_segment_beats: int = _OCTAVE_MIN_SEGMENT_BEATS,
) -> tuple[list[float], list[float]]:
    """Keep one continuous metrical level across a beat track."""
    if not 0 < tolerance < 1:
        raise ValueError("octave tolerance must be between 0 and 1")
    if min_segment_beats < 1:
        raise ValueError("minimum octave segment length must be positive")
    beat_times = [float(value) for value in beats]
    downbeat_times = [float(value) for value in downbeats]
    if len(beat_times) < min_segment_beats + 2:
        return beat_times, downbeat_times

    intervals = [right - left for left, right in zip(beat_times, beat_times[1:])]
    if any(interval <= 0 for interval in intervals):
        return beat_times, downbeat_times

    global_interval = median(intervals)
    states = _classify_octaves(intervals, global_interval, tolerance)
    states = _remove_short_corrections(states, min_segment_beats)
    if all(state == 1.0 for state in states):
        return beat_times, downbeat_times

    corrected_beats, first_changed_time = _apply_corrections(
        beat_times, states, global_interval
    )
    corrected_downbeats = _rebuild_downbeats(
        beat_times,
        downbeat_times,
        corrected_beats,
        first_changed_time,
    )
    return corrected_beats, corrected_downbeats


def _classify_octaves(
    intervals: list[float],
    global_interval: float,
    tolerance: float,
) -> list[float]:
    allowed_change = log2(1 + tolerance)
    costs: list[dict[float, float]] = []
    previous_states: list[dict[float, float]] = []

    first_costs: dict[float, float] = {}
    first_previous: dict[float, float] = {}
    for factor in _OCTAVE_FACTORS:
        candidate = intervals[0] * factor
        first_costs[factor] = _global_cost(candidate, global_interval, allowed_change)
        first_costs[factor] += _correction_cost(factor)
        first_previous[factor] = factor
    costs.append(first_costs)
    previous_states.append(first_previous)

    for index in range(1, len(intervals)):
        current_costs: dict[float, float] = {}
        current_previous: dict[float, float] = {}
        for factor in _OCTAVE_FACTORS:
            candidate = intervals[index] * factor
            emission = _global_cost(candidate, global_interval, allowed_change)
            emission += _correction_cost(factor)
            options: list[tuple[float, float]] = []
            for previous_factor in _OCTAVE_FACTORS:
                previous_candidate = intervals[index - 1] * previous_factor
                change = abs(log2(candidate / previous_candidate))
                smoothness = 8 * max(0.0, change - allowed_change) ** 2
                transition = 0.35 if previous_factor != factor else 0.0
                options.append(
                    (
                        costs[index - 1][previous_factor]
                        + emission
                        + smoothness
                        + transition,
                        previous_factor,
                    )
                )
            best_cost, best_previous = min(options, key=lambda option: option[0])
            current_costs[factor] = best_cost
            current_previous[factor] = best_previous
        costs.append(current_costs)
        previous_states.append(current_previous)

    state = min(costs[-1], key=costs[-1].get)
    result = [state]
    for index in range(len(intervals) - 1, 0, -1):
        state = previous_states[index][state]
        result.append(state)
    result.reverse()
    return result


def _global_cost(candidate: float, reference: float, allowed_change: float) -> float:
    distance = abs(log2(candidate / reference))
    return 2 * max(0.0, distance - allowed_change) ** 2


def _correction_cost(factor: float) -> float:
    return 0.0 if factor == 1.0 else 0.04


def _remove_short_corrections(states: list[float], min_segment_beats: int) -> list[float]:
    result = states.copy()
    start = 0
    while start < len(result):
        end = start + 1
        while end < len(result) and result[end] == result[start]:
            end += 1
        if result[start] != 1.0 and end - start < min_segment_beats:
            result[start:end] = [1.0] * (end - start)
        start = end
    return result


def _apply_corrections(
    beats: list[float],
    states: list[float],
    global_interval: float,
) -> tuple[list[float], float]:
    corrected = [beats[0]]
    first_changed_time = beats[-1]
    index = 0
    while index < len(states):
        state = states[index]
        if state == 1.0:
            corrected.append(beats[index + 1])
            index += 1
            continue

        end = index + 1
        while end < len(states) and states[end] == state:
            end += 1
        first_changed_time = min(first_changed_time, beats[index])
        if state == 0.5:
            for interval_index in range(index, end):
                left = beats[interval_index]
                right = beats[interval_index + 1]
                corrected.extend(((left + right) / 2, right))
        else:
            corrected = _collapse_double_time_segment(
                corrected,
                beats[index : end + 1],
                beats[end + 1] if end + 1 < len(beats) else None,
                global_interval,
            )
        index = end
    return _deduplicate(corrected), first_changed_time


def _collapse_double_time_segment(
    prefix: list[float],
    segment: list[float],
    next_beat: float | None,
    global_interval: float,
) -> list[float]:
    keep_boundary = prefix + segment[2::2]
    shift_boundary = prefix[:-1] + segment[1::2]
    expected = _local_interval(prefix, global_interval)
    keep_cost = _boundary_cost(keep_boundary, next_beat, expected)
    shift_cost = _boundary_cost(shift_boundary, next_beat, expected)
    return keep_boundary if keep_cost <= shift_cost else shift_boundary


def _local_interval(beats: list[float], fallback: float) -> float:
    if len(beats) < 2:
        return fallback
    sample = beats[-6:]
    intervals = [right - left for left, right in zip(sample, sample[1:])]
    intervals = [interval for interval in intervals if interval > 0]
    return median(intervals) if intervals else fallback


def _boundary_cost(
    beats: list[float],
    next_beat: float | None,
    expected: float,
) -> float:
    sample = beats[-7:].copy()
    if next_beat is not None and (not sample or next_beat > sample[-1]):
        sample.append(next_beat)
    intervals = [right - left for left, right in zip(sample, sample[1:])]
    if not intervals or any(interval <= 0 for interval in intervals):
        return float("inf")
    continuity = sum(abs(log2(interval / expected)) for interval in intervals)
    smoothness = sum(
        abs(log2(right / left)) for left, right in zip(intervals, intervals[1:])
    )
    return continuity + 2 * smoothness


def _deduplicate(beats: list[float]) -> list[float]:
    result: list[float] = []
    for beat in beats:
        if not result or beat > result[-1]:
            result.append(beat)
    return result


def _rebuild_downbeats(
    original_beats: list[float],
    original_downbeats: list[float],
    corrected_beats: list[float],
    first_changed_time: float,
) -> list[float]:
    if not original_downbeats or len(corrected_beats) < 2:
        return []
    meter = _infer_meter(original_beats, original_downbeats)
    if meter is None:
        return [
            corrected_beats[_nearest_index(corrected_beats, downbeat)]
            for downbeat in original_downbeats
        ]

    reliable_downbeats = [
        downbeat for downbeat in original_downbeats if downbeat < first_changed_time
    ]
    phase_sources = reliable_downbeats or original_downbeats[:1]
    phases = [
        _nearest_index(corrected_beats, downbeat) % meter for downbeat in phase_sources
    ]
    phase = Counter(phases).most_common(1)[0][0]
    return [
        beat for index, beat in enumerate(corrected_beats) if index % meter == phase
    ]


def _infer_meter(beats: list[float], downbeats: list[float]) -> int | None:
    indices = [_nearest_index(beats, downbeat) for downbeat in downbeats]
    differences = [
        right - left
        for left, right in zip(indices, indices[1:])
        if 2 <= right - left <= 12
    ]
    if not differences:
        return None
    return Counter(differences).most_common(1)[0][0]


def _nearest_index(values: list[float], target: float) -> int:
    index = bisect_left(values, target)
    if index == 0:
        return 0
    if index == len(values):
        return len(values) - 1
    before = values[index - 1]
    after = values[index]
    return index - 1 if target - before <= after - target else index


SUPPORTED_EXTENSIONS = frozenset({".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"})
DEFAULT_INPUT = DEVELOP_DATA_DIR / "audio"
DEFAULT_OUTPUT = DEVELOP_DATA_DIR / "beats"
TEMP_DIR = PROJECT_ROOT / "temp"
logger = get_logger("beat_detection")


def find_audio_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _select_device(torch: Any, requested: str) -> tuple[str, bool]:
    if requested == "auto":
        return ("cuda", True) if torch.cuda.is_available() else ("cpu", False)
    if requested == "cpu":
        return "cpu", False
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Beat This requested CUDA, but no CUDA device is available")
        device = torch.device(requested)
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        return f"cuda:{index}", True
    raise ValueError("Device must be auto, cpu, cuda, or cuda:N")


def _decode_with_ffmpeg(audio_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(f"Could not decode {audio_path}; FFmpeg is not available on PATH")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=TEMP_DIR, delete=False) as file:
        decoded_path = Path(file.name)
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(decoded_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        decoded_path.unlink(missing_ok=True)
        detail = process.stderr.strip() or f"FFmpeg exited with code {process.returncode}"
        raise RuntimeError(f"Could not decode {audio_path}: {detail}")
    return decoded_path


class BeatDetector:
    """Load one Beat This model and reuse it for a batch."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dbn: bool = False,
    ) -> None:
        try:
            import torch
            from beat_this.inference import File2Beats
        except ImportError as exc:
            raise RuntimeError("Beat This is not installed; run `uv sync`") from exc

        self.device, float16 = _select_device(torch, device)
        try:
            self.detector = File2Beats(
                checkpoint_path=model_name,
                device=self.device,
                float16=float16,
                dbn=dbn,
            )
        except ImportError as exc:
            if dbn:
                raise RuntimeError(
                    "Beat This DBN post-processing requires the madmom dependency"
                ) from exc
            raise

    def detect(self, audio_path: Path) -> tuple[Any, Any]:
        try:
            result = self.detector(str(audio_path))
        except RuntimeError as exc:
            if "Could not load audio" not in str(exc):
                raise
            decoded_path = _decode_with_ffmpeg(audio_path)
            try:
                result = self.detector(str(decoded_path))
            finally:
                decoded_path.unlink(missing_ok=True)
        if not _OCTAVE_CORRECTION:
            return result
        return _normalize_beat_octaves(
            *result,
            tolerance=_OCTAVE_TOLERANCE,
            min_segment_beats=_OCTAVE_MIN_SEGMENT_BEATS,
        )

    def release(self) -> None:
        self.detector = None


def write_result(
    detector: BeatDetector,
    source: Path,
    destination: Path,
    audio_reference: str | None = None,
) -> None:
    beats, downbeats = detector.detect(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "audio": audio_reference or str(source),
                "beats": [float(value) for value in beats],
                "downbeats": [float(value) for value in downbeats],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def detect_path(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    dbn: bool,
) -> int:
    files = find_audio_files(input_path)
    if not files:
        logger.warning("No supported audio files found in %s", input_path)
        return 0
    root = input_path if input_path.is_dir() else input_path.parent
    detector = BeatDetector(model_name, device, dbn)
    failures = 0
    for source in files:
        destination = (output_dir / source.relative_to(root)).with_suffix(".json")
        try:
            write_result(detector, source, destination)
            logger.info("Wrote: %s", destination)
        except Exception as exc:  # keep processing remaining files
            failures += 1
            logger.error("Failed %s: %s", source, exc)
    return failures


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Detect beats and downbeats with Beat This.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dbn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config).require_beat_detection()
    enabled = config.enabled if args.enabled is None else args.enabled
    if not enabled:
        logger.info("Beat detection is disabled by configuration")
        return 0
    try:
        return 1 if detect_path(
            args.input,
            args.output,
            args.model or config.model,
            args.device or config.device,
            config.dbn if args.dbn is None else args.dbn,
        ) else 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

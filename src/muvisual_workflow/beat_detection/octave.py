"""Shared beat-level octave correction."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from math import log2
from statistics import median
from typing import Iterable

_OCTAVE_TOLERANCE = 0.35
_OCTAVE_MIN_SEGMENT_BEATS = 4
_OCTAVE_FACTORS = (0.5, 1.0, 2.0)
def normalize_beat_octaves(
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


def infer_time_signature(beats: list[float], downbeats: list[float]) -> str | None:
    """Infer a quarter-note time signature from beats between downbeats."""
    meter = _infer_meter(beats, downbeats)
    return f"{meter}/4" if meter is not None else None


def _nearest_index(values: list[float], target: float) -> int:
    index = bisect_left(values, target)
    if index == 0:
        return 0
    if index == len(values):
        return len(values) - 1
    before = values[index - 1]
    after = values[index]
    return index - 1 if target - before <= after - target else index

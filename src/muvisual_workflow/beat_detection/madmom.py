"""Beat and downbeat estimation with madmom."""

from __future__ import annotations

import collections
import collections.abc
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class MadmomBeatResult:
    beats: list[float]
    downbeats: list[float]
    bpm: float
    raw_bpm: float
    doubled_bpm: bool
    model: str = "madmom RNNBeatProcessor + DBNBeatTrackingProcessor"


def _compatibility() -> None:
    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence
    for name, value in (("float", np.float64), ("int", np.int_), ("complex", np.complex128), ("bool", np.bool_)):
        if name not in np.__dict__:
            setattr(np, name, value)


def _bpm(beats: list[float], fallback: float = 120.0) -> float:
    if len(beats) < 2:
        return fallback
    intervals = np.diff(np.asarray(beats, dtype=np.float64))
    intervals = intervals[intervals > 0]
    return 60.0 / float(np.median(intervals)) if len(intervals) else fallback


def _insert_midpoints(beats: list[float]) -> list[float]:
    if len(beats) < 2:
        return beats.copy()
    result: list[float] = []
    for left, right in zip(beats, beats[1:]):
        result.extend((left, left + (right - left) * 0.5))
    result.append(beats[-1])
    return result


class MadmomBeatDetector:
    """Run madmom and correct DBN output that is one octave too slow."""

    def __init__(self, minimum_bpm: float = 90.0, maximum_bpm: float = 180.0) -> None:
        if minimum_bpm <= 0 or maximum_bpm <= minimum_bpm:
            raise ValueError("Invalid madmom BPM range")
        self.minimum_bpm = minimum_bpm
        self.maximum_bpm = maximum_bpm

    def detect_result(self, audio_path: Path) -> MadmomBeatResult:
        _compatibility()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
            try:
                from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
            except ImportError as exc:
                raise RuntimeError("madmom is not installed; run `uv sync`") from exc
        activations = RNNBeatProcessor()(str(audio_path))
        raw_beats = [float(value) for value in DBNBeatTrackingProcessor(fps=100)(activations)]
        raw_bpm = _bpm(raw_beats)
        doubled = raw_bpm < self.minimum_bpm and raw_bpm * 2 <= self.maximum_bpm
        beats = _insert_midpoints(raw_beats) if doubled else raw_beats
        return MadmomBeatResult(
            beats=beats,
            downbeats=beats[::4],
            bpm=_bpm(beats, raw_bpm),
            raw_bpm=raw_bpm,
            doubled_bpm=doubled,
        )

    def detect(self, audio_path: Path) -> tuple[list[float], list[float]]:
        result = self.detect_result(audio_path)
        return result.beats, result.downbeats

    def release(self) -> None:
        return None

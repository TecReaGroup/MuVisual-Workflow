"""Beat and downbeat estimation with madmom."""

from __future__ import annotations

import collections
import collections.abc
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from muvisual_workflow.beat_detection.octave import normalize_beat_octaves


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


class MadmomBeatDetector:
    """Run madmom and normalize transient half-time or double-time segments."""

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
        processor = DBNBeatTrackingProcessor(
            min_bpm=self.minimum_bpm,
            max_bpm=self.maximum_bpm,
            fps=100,
        )
        raw_beats = [float(value) for value in processor(activations)]
        raw_downbeats = raw_beats[::4]
        raw_bpm = _bpm(raw_beats)
        beats, downbeats = normalize_beat_octaves(raw_beats, raw_downbeats)
        bpm = _bpm(beats, raw_bpm)
        if bpm > self.maximum_bpm:
            beats, downbeats, bpm = raw_beats, raw_downbeats, raw_bpm
        doubled = bpm > raw_bpm * 1.5
        return MadmomBeatResult(
            beats=beats,
            downbeats=downbeats,
            bpm=bpm,
            raw_bpm=raw_bpm,
            doubled_bpm=doubled,
        )

    def detect(self, audio_path: Path) -> tuple[list[float], list[float]]:
        result = self.detect_result(audio_path)
        return result.beats, result.downbeats

    def release(self) -> None:
        return None

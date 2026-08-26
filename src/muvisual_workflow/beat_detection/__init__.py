"""Beat and downbeat detection workflow step."""

from muvisual_workflow.beat_detection.beat_this import BeatDetector, main, write_result
from muvisual_workflow.beat_detection.madmom import MadmomBeatDetector, MadmomBeatResult

__all__ = ["BeatDetector", "MadmomBeatDetector", "MadmomBeatResult", "main", "write_result"]

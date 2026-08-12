"""MuScriptor model implementation for the audio-to-MIDI step."""

from __future__ import annotations

import io
import sys
import wave
from contextlib import redirect_stdout
from contextvars import ContextVar
from pathlib import Path
from typing import TextIO

_UNFINISHED_NOTE_DURATION_SECONDS = 5.0
_audio_duration_seconds: ContextVar[float | None] = ContextVar(
    "muscriptor_audio_duration_seconds",
    default=None,
)


def _install_open_note_finish_patch() -> None:
    from muscriptor.events import OpenNoteTracker, _EndNote, _NoteAction
    from muscriptor.tokenizer.notes import MINIMUM_NOTE_DURATION_SEC

    if getattr(OpenNoteTracker.finish, "_muvisual_patched", False):
        return

    original_finish = OpenNoteTracker.finish

    def finish_with_audio_end(self: OpenNoteTracker) -> list[_NoteAction]:
        audio_duration = _audio_duration_seconds.get()
        if audio_duration is None:
            return original_finish(self)

        # Preserve MuScriptor's handling of a malformed final chunk.
        if self._chunk_started and self._in_prologue:
            return self._end_all(self._seek_time)

        actions: list[_NoteAction] = [
            _EndNote(
                *key,
                max(
                    onset + MINIMUM_NOTE_DURATION_SEC,
                    min(onset + _UNFINISHED_NOTE_DURATION_SECONDS, audio_duration),
                ),
            )
            for key, onset in self._open.items()
        ]
        self._open.clear()
        return actions

    finish_with_audio_end._muvisual_patched = True
    OpenNoteTracker.finish = finish_with_audio_end


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        frame_rate = audio.getframerate()
        if frame_rate <= 0:
            raise RuntimeError(f"Invalid WAV sample rate: {path}")
        return audio.getnframes() / frame_rate


class _MuscriptorOutputFilter(io.TextIOBase):
    """Suppress MuScriptor's per-segment timing output."""

    def __init__(self, target: TextIO) -> None:
        self._target = target
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.startswith("[muscriptor]"):
                self._target.write(f"{line}\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer and not self._buffer.startswith("[muscriptor]"):
            self._target.write(self._buffer)
        self._buffer = ""
        self._target.flush()


class MuscriptorModel:
    """Load one MuScriptor model and reuse it for a batch of files."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str | None = None,
        instruments: tuple[str, ...] = ("acoustic_piano",),
    ) -> None:
        try:
            import torch
            from muscriptor import TranscriptionModel
        except ImportError as exc:
            raise RuntimeError("MuScriptor is not installed; run `uv sync`") from exc

        _install_open_note_finish_patch()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("MuScriptor requested CUDA, but no CUDA device is available")
        if dtype in {None, "auto"}:
            dtype = "float16" if device.startswith("cuda") else "float32"

        self.device = device
        self.dtype = dtype
        self.instruments = list(instruments)
        self.model = TranscriptionModel.load_model(model_name, device=device, dtype=dtype)

    def transcribe(self, input_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_token = _audio_duration_seconds.set(_wav_duration_seconds(input_path))
        try:
            with redirect_stdout(_MuscriptorOutputFilter(sys.stdout)):
                midi_bytes = self.model.transcribe_to_midi(
                    input_path,
                    instruments=self.instruments,
                )
        finally:
            _audio_duration_seconds.reset(duration_token)
        output_path.write_bytes(midi_bytes)

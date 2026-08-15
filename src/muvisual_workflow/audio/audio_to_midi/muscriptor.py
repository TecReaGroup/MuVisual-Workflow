"""MuScriptor model implementation for the audio-to-MIDI step."""

from __future__ import annotations

import io
import sys
import wave
from contextlib import redirect_stdout
from pathlib import Path
from typing import TextIO



def _audio_duration_seconds(source: Path) -> float:
    try:
        with wave.open(str(source), "rb") as audio:
            frame_rate = audio.getframerate()
            if frame_rate <= 0:
                raise RuntimeError(f"Invalid WAV sample rate: {source}")
            return audio.getnframes() / frame_rate
    except (OSError, wave.Error):
        try:
            import mutagen
        except ImportError as exc:
            raise RuntimeError("Mutagen is not installed; run `uv sync`") from exc

        try:
            audio = mutagen.File(source)
        except mutagen.MutagenError as exc:
            raise RuntimeError(f"Could not read audio duration: {source}: {exc}") from exc
        if audio is None or audio.info is None:
            raise RuntimeError(f"Could not read audio duration: {source}")
        return float(audio.info.length)


def _trim_midi_bytes(midi_bytes: bytes, duration_seconds: float) -> bytes:
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("Mido is not installed; run `uv sync`") from exc

    midi = mido.MidiFile(file=io.BytesIO(midi_bytes))
    tempo = next(
        (
            message.tempo
            for track in midi.tracks
            for message in track
            if message.type == "set_tempo"
        ),
        500_000,
    )
    audio_end_tick = round(
        mido.second2tick(duration_seconds, midi.ticks_per_beat, tempo)
    )

    for track in midi.tracks:
        events = []
        active_notes: dict[tuple[int, int], int] = {}
        absolute_time = 0
        event_order = 0
        for message in track:
            absolute_time += message.time
            if message.type == "end_of_track":
                continue

            is_note_start = message.type == "note_on" and message.velocity > 0
            is_note_end = message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            )

            if is_note_start:
                if absolute_time >= audio_end_tick:
                    continue
                note_key = (message.channel, message.note)
                active_notes[note_key] = active_notes.get(note_key, 0) + 1
            elif is_note_end:
                if absolute_time > audio_end_tick:
                    continue
                note_key = (message.channel, message.note)
                active_count = active_notes.get(note_key, 0)
                if active_count > 1:
                    active_notes[note_key] = active_count - 1
                elif active_count == 1:
                    del active_notes[note_key]
            elif absolute_time > audio_end_tick:
                continue

            events.append((absolute_time, event_order, message.copy(time=0)))
            event_order += 1

        for (channel, note), count in active_notes.items():
            for _ in range(count):
                events.append(
                    (
                        audio_end_tick,
                        event_order,
                        mido.Message(
                            "note_off",
                            channel=channel,
                            note=note,
                            velocity=0,
                            time=0,
                        ),
                    )
                )
                event_order += 1

        events.append(
            (
                audio_end_tick,
                event_order,
                mido.MetaMessage("end_of_track", time=0),
            )
        )
        events.sort(key=lambda event: (event[0], event[1]))

        rebuilt = mido.MidiTrack()
        previous_time = 0
        for event_time, _, message in events:
            rebuilt.append(message.copy(time=event_time - previous_time))
            previous_time = event_time
        track.clear()
        track.extend(rebuilt)

    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


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
        with redirect_stdout(_MuscriptorOutputFilter(sys.stdout)):
            midi_bytes = self.model.transcribe_to_midi(
                input_path,
                instruments=self.instruments,
            )
        duration_seconds = _audio_duration_seconds(input_path)
        output_path.write_bytes(_trim_midi_bytes(midi_bytes, duration_seconds))

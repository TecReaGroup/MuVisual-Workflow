"""Audio format conversion helpers."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def convert_to_mp3(source: Path, destination: Path) -> Path:
    """Copy an MP3 source or encode another audio format as MP3."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.casefold() == ".mp3":
        shutil.copy2(source, destination)
        return destination

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP3 output requires FFmpeg, but ffmpeg was not found on PATH")
    process = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-map_metadata",
            "0",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(destination),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = process.stderr.strip() or f"FFmpeg exited with code {process.returncode}"
        raise RuntimeError(f"Could not encode {source} as MP3: {detail}")
    return destination



def convert_to_wav(source: Path, destination: Path) -> Path:
    """Copy a WAV source or decode another audio format as PCM WAV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.casefold() == ".wav":
        shutil.copy2(source, destination)
        return destination

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("WAV output requires FFmpeg, but ffmpeg was not found on PATH")
    conversion = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if conversion.returncode != 0:
        destination.unlink(missing_ok=True)
        detail = conversion.stderr.strip() or f"FFmpeg exited with code {conversion.returncode}"
        raise RuntimeError(f"Could not decode {source} as WAV: {detail}")
    return destination

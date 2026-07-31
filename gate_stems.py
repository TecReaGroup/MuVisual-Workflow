"""Apply a configurable noise gate to every WAV in data/stem.

Examples:
    python gate_stems.py
    python gate_stems.py --threshold-db -48 --attack-ms 10 --hold-ms 80 --release-ms 150

The gate is driven by short-time RMS level. Quiet sections are attenuated to
zero; attack, hold, and release are applied to the gain envelope rather than
cutting individual samples abruptly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "data" / "stem"
DEFAULT_OUTPUT = ROOT / "data" / "stem_gated"
DEFAULT_THRESHOLD_DB = -48.0
DEFAULT_ATTACK_MS = 8.0
DEFAULT_HOLD_MS = 80.0
DEFAULT_RELEASE_MS = 180.0
DEFAULT_ANALYSIS_MS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noise-gate stem WAV files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold-db", type=float, default=DEFAULT_THRESHOLD_DB,
                        help=f"Gate threshold in dBFS (default: {DEFAULT_THRESHOLD_DB:g})")
    parser.add_argument("--attack-ms", type=float, default=DEFAULT_ATTACK_MS,
                        help=f"Time to open the gate (default: {DEFAULT_ATTACK_MS:g} ms)")
    parser.add_argument("--hold-ms", type=float, default=DEFAULT_HOLD_MS,
                        help=f"Minimum open time after level falls below threshold (default: {DEFAULT_HOLD_MS:g} ms)")
    parser.add_argument("--release-ms", type=float, default=DEFAULT_RELEASE_MS,
                        help=f"Time to close the gate (default: {DEFAULT_RELEASE_MS:g} ms)")
    parser.add_argument("--analysis-ms", type=float, default=DEFAULT_ANALYSIS_MS,
                        help=f"RMS analysis block size (default: {DEFAULT_ANALYSIS_MS:g} ms)")
    return parser.parse_args()


def to_float32(data: np.ndarray) -> tuple[np.ndarray, float]:
    if np.issubdtype(data.dtype, np.integer):
        peak = float(np.iinfo(data.dtype).max)
        return data.astype(np.float32) / peak, peak
    return data.astype(np.float32), 1.0


def restore_dtype(data: np.ndarray, dtype: np.dtype, peak: float) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(data * peak), info.min, info.max).astype(dtype)
    return data.astype(dtype)


def gate_audio(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    attack_ms: float,
    hold_ms: float,
    release_ms: float,
    analysis_ms: float,
) -> np.ndarray:
    mono_or_stereo = audio if audio.ndim == 2 else audio[:, None]
    samples = len(mono_or_stereo)
    block = max(1, round(sample_rate * analysis_ms / 1000))
    count = (samples + block - 1) // block
    levels = np.empty(count, dtype=np.float32)
    for index in range(count):
        chunk = mono_or_stereo[index * block:min(samples, (index + 1) * block)]
        levels[index] = np.sqrt(np.mean(chunk * chunk))

    threshold = 10 ** (threshold_db / 20)
    attack_blocks = max(1, round(attack_ms / analysis_ms))
    hold_blocks = max(0, round(hold_ms / analysis_ms))
    release_blocks = max(1, round(release_ms / analysis_ms))
    gain = np.zeros(count, dtype=np.float32)
    current = 0.0
    hold_left = 0
    for index, level in enumerate(levels):
        if level >= threshold:
            hold_left = hold_blocks
            current = min(1.0, current + 1.0 / attack_blocks)
        elif hold_left:
            hold_left -= 1
        else:
            current = max(0.0, current - 1.0 / release_blocks)
        gain[index] = current

    centers = np.minimum(np.arange(count) * block + block / 2, samples - 1)
    sample_gain = np.interp(np.arange(samples), centers, gain).astype(np.float32)
    return audio * (sample_gain[:, None] if audio.ndim == 2 else sample_gain)


def main() -> None:
    args = parse_args()
    if args.attack_ms < 0 or args.hold_ms < 0 or args.release_ms <= 0 or args.analysis_ms <= 0:
        raise SystemExit("attack/hold/analysis must be >= 0 and release must be > 0")
    files = sorted(args.input.glob("*.wav"))
    if not files:
        raise SystemExit(f"No WAV files found in {args.input.resolve()}")
    args.output.mkdir(parents=True, exist_ok=True)
    for source in files:
        sample_rate, raw = wavfile.read(source)
        audio, peak = to_float32(raw)
        gated = gate_audio(
            audio, sample_rate, args.threshold_db, args.attack_ms,
            args.hold_ms, args.release_ms, args.analysis_ms,
        )
        destination = args.output / source.name
        wavfile.write(destination, sample_rate, restore_dtype(gated, raw.dtype, peak))
        print(f"{source.name} -> {destination}")


if __name__ == "__main__":
    main()

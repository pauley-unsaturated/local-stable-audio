#!/usr/bin/env python3
"""Objective sanity checks on a generated WAV, for when nobody is around to
listen: duration, level, spectral centroid, silence fraction, and a crude
tempo estimate from spectral-flux onset autocorrelation. Not a substitute for
ears; Gate 0 still says "sounds like the prompt".

    python3 tools/wavcheck.py out.wav [--bpm-hint 128]
"""
from __future__ import annotations

import argparse
import sys
import wave

import numpy as np


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        nch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        sys.exit(f"unsupported sample width {sw}")
    return x.reshape(-1, nch).T, sr


def stft_mag(x: np.ndarray, n: int = 2048, hop: int = 512) -> np.ndarray:
    win = np.hanning(n).astype(np.float32)
    frames = 1 + max(0, (len(x) - n) // hop)
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * win, axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--bpm-hint", type=float, default=None)
    a = ap.parse_args()
    x, sr = read_wav(a.path)
    nch, n = x.shape
    dur = n / sr
    mono = x.mean(axis=0)
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x**2)))
    # silence: 50 ms blocks below -60 dBFS
    blk = int(sr * 0.05)
    blocks = mono[: (n // blk) * blk].reshape(-1, blk)
    brms = np.sqrt(np.mean(blocks**2, axis=1))
    silent = float(np.mean(brms < 10 ** (-60 / 20)))
    # spectral centroid (mean over frames)
    S = stft_mag(mono)
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    cent = float(np.mean((S * freqs).sum(axis=1) / np.maximum(S.sum(axis=1), 1e-9)))
    # stereo width: correlation between channels
    corr = float(np.corrcoef(x[0], x[1])[0, 1]) if nch == 2 else 1.0
    # tempo: spectral flux → autocorrelation over 60–200 BPM
    hop = 512
    flux = np.maximum(np.diff(np.log1p(S), axis=0), 0).sum(axis=1)
    flux -= flux.mean()
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = sr / hop
    lags = np.arange(len(ac))
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 60)
    seg = ac[lo:hi]
    best = lo + int(np.argmax(seg))
    bpm = 60 * fps / best
    top = sorted(((60 * fps / (lo + i), seg[i]) for i in np.argsort(seg)[-5:]), key=lambda t: -t[1])
    print(f"file      {a.path}")
    print(f"format    {nch} ch, {sr} Hz, {dur:.3f} s ({n} samples)")
    print(f"level     peak {20*np.log10(max(peak,1e-9)):.1f} dBFS   rms {20*np.log10(max(rms,1e-9)):.1f} dBFS   crest {20*np.log10(max(peak/max(rms,1e-9),1e-9)):.1f} dB")
    print(f"silence   {100*silent:.1f}% of 50 ms blocks below -60 dBFS")
    print(f"spectrum  mean centroid {cent:.0f} Hz")
    print(f"stereo    L/R correlation {corr:.2f}")
    print(f"tempo     best {bpm:.1f} BPM; candidates " + ", ".join(f"{b:.0f}" for b, _ in top))
    if a.bpm_hint:
        cands = [bpm, bpm * 2, bpm / 2]
        close = min(abs(c - a.bpm_hint) for c in cands)
        print(f"          vs hint {a.bpm_hint:.0f}: nearest (incl. octave) off by {close:.1f} BPM")
    verdict = "OK" if (peak > 0.05 and silent < 0.5 and not np.isnan(cent)) else "SUSPECT"
    print(f"verdict   {verdict} (non-silent, sane level, finite spectrum)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

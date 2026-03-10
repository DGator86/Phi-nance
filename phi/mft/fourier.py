"""Fourier-domain Market Field Theory (MFT) indicators."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rolling_spectral_power(series: pd.Series, window: int, bands: Sequence[tuple[float, float]]) -> pd.DataFrame:
    """Return rolling relative spectral power in frequency bands.

    Band limits are expressed as fractions of Nyquist in [0, 1].
    """
    values = series.astype(float).ffill().bfill().fillna(0.0).to_numpy()
    n = len(values)
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")

    freqs = np.fft.rfftfreq(w, d=1.0)
    nyquist = 0.5
    frac_freqs = freqs / nyquist

    band_labels = [f"band_{i}" for i in range(len(bands))]
    out = np.full((n, len(bands)), np.nan, dtype=float)

    for i in range(w - 1, n):
        windowed = values[i - w + 1 : i + 1]
        spectrum = np.fft.rfft(windowed - np.mean(windowed))
        power = np.abs(spectrum) ** 2
        total_power = float(power.sum())
        if total_power <= 0:
            out[i, :] = 0.0
            continue

        for j, (low, high) in enumerate(bands):
            low_f = max(0.0, float(low))
            high_f = min(1.0, float(high))
            mask = (frac_freqs >= low_f) & (frac_freqs < high_f)
            out[i, j] = float(power[mask].sum() / total_power) if np.any(mask) else 0.0

    return pd.DataFrame(out, index=series.index, columns=band_labels)


def rolling_spectral_centroid(series: pd.Series, window: int) -> pd.Series:
    """Return rolling spectral centroid as a fraction of Nyquist."""
    values = series.astype(float).ffill().bfill().fillna(0.0).to_numpy()
    n = len(values)
    w = int(window)
    freqs = np.fft.rfftfreq(w, d=1.0)
    nyquist = 0.5
    frac_freqs = freqs / nyquist

    out = np.full(n, np.nan, dtype=float)
    for i in range(w - 1, n):
        windowed = values[i - w + 1 : i + 1]
        spectrum = np.fft.rfft(windowed - np.mean(windowed))
        power = np.abs(spectrum) ** 2
        denom = float(power.sum())
        out[i] = 0.0 if denom <= 0 else float((frac_freqs * power).sum() / denom)

    return pd.Series(out, index=series.index, name="spectral_centroid")


def rolling_phase_coherence(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    """Return rolling phase coherence from normalized cross-spectrum."""
    a = series_a.astype(float).ffill().bfill().fillna(0.0).to_numpy()
    b = series_b.astype(float).ffill().bfill().fillna(0.0).to_numpy()
    n = len(a)
    w = int(window)
    out = np.full(n, np.nan, dtype=float)

    for i in range(w - 1, n):
        aw = a[i - w + 1 : i + 1] - np.mean(a[i - w + 1 : i + 1])
        bw = b[i - w + 1 : i + 1] - np.mean(b[i - w + 1 : i + 1])
        fa = np.fft.rfft(aw)
        fb = np.fft.rfft(bw)
        cross = fa * np.conj(fb)
        denom = np.abs(cross)
        valid = denom > 0
        if not np.any(valid):
            out[i] = 0.0
            continue
        normalized = cross[valid] / denom[valid]
        out[i] = float(np.abs(np.mean(normalized)))

    return pd.Series(out, index=series_a.index, name="phase_coherence").clip(0.0, 1.0)

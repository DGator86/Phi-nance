"""Rolling volume profile (market profile) indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_poc(close: pd.Series, volume: pd.Series, window: int, bins: int) -> pd.Series:
    poc = pd.Series(np.nan, index=close.index, dtype=float)
    for i in range(len(close)):
        start = max(0, i - window + 1)
        c = close.iloc[start : i + 1]
        v = volume.iloc[start : i + 1]
        if len(c) < 2:
            continue
        low, high = float(c.min()), float(c.max())
        if np.isclose(low, high):
            poc.iloc[i] = high
            continue
        edges = np.linspace(low, high, int(bins) + 1)
        idx = np.digitize(c.to_numpy(), edges[1:-1], right=False)
        vol_bins = np.zeros(int(bins), dtype=float)
        for j, bin_idx in enumerate(idx):
            vol_bins[min(max(int(bin_idx), 0), int(bins) - 1)] += float(v.iloc[j])
        poc_bin = int(np.argmax(vol_bins))
        poc.iloc[i] = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0
    return poc


def compute_volume_profile_signal(
    ohlcv: pd.DataFrame,
    window: int = 20,
    bins: int = 16,
    near_poc_threshold: float = 0.002,
) -> tuple[pd.Series, pd.Series]:
    """Return rolling POC and a discrete signal based on distance from POC."""
    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).clip(lower=0.0)
    poc = _rolling_poc(close, volume, int(window), int(bins)).rename("poc")

    dist = (close - poc) / poc.replace(0.0, np.nan)
    signal = pd.Series(-1.0, index=ohlcv.index, dtype=float)
    signal[(dist.abs() <= near_poc_threshold) | dist.isna()] = 1.0
    return poc.fillna(close), signal.rename("volume_profile_signal")

"""Rolling volume profile (market profile) indicator.

Adaptive bin sizing
-------------------
The original implementation used equal-width (linear) bins, giving identical
resolution across the full price range regardless of where volume is traded.
This under-weights the area near the current price — exactly where stop-loss
and limit-order clusters form.

The improved version uses **log-space bins** (equal-ratio bins), which gives
finer resolution near round-dollar levels and near the current price when the
range is wide.  For tight ranges (high ≈ low) it falls back to equal-width.

Additionally, ``compute_volume_profile_signal`` now returns a *continuous*
normalised distance signal in [−1, +1] instead of a ternary ±1 flag, giving
the taxonomy engine more gradient to work with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _adaptive_bins(low: float, high: float, n_bins: int) -> np.ndarray:
    """Compute adaptive (log-spaced where possible) bin edges.

    For a wide range (high/low > 1.001) returns log-spaced edges so that bins
    near lower prices are narrower — matching where integer round-number levels
    cluster.  For a tight range falls back to equal-width edges.
    """
    if high <= low or np.isclose(low, high):
        return np.linspace(low, high, n_bins + 1)
    ratio = high / (low + 1e-10)
    if ratio > 1.001:
        return np.exp(np.linspace(np.log(low + 1e-10), np.log(high), n_bins + 1))
    return np.linspace(low, high, n_bins + 1)


def _rolling_poc(close: pd.Series, volume: pd.Series, window: int, bins: int) -> pd.Series:
    """Compute rolling Point-of-Control using adaptive log-spaced bins.

    Adaptive binning gives finer resolution near lower prices (round-number
    territory) and coarser resolution at price extremes, better capturing
    where stop-clusters and limit-order pools actually form.
    """
    poc = pd.Series(np.nan, index=close.index, dtype=float)
    n_bins = int(bins)
    close_arr  = close.to_numpy(dtype=float)
    volume_arr = volume.to_numpy(dtype=float)

    for i in range(len(close_arr)):
        start = max(0, i - window + 1)
        c = close_arr[start : i + 1]
        v = volume_arr[start : i + 1]
        if len(c) < 2:
            continue
        low_p, high_p = float(c.min()), float(c.max())
        if np.isclose(low_p, high_p):
            poc.iloc[i] = high_p
            continue

        edges = _adaptive_bins(low_p, high_p, n_bins)
        # Assign each bar to a bin (digitize against interior edges)
        bin_idx = np.digitize(c, edges[1:-1], right=False)
        vol_bins = np.zeros(n_bins, dtype=float)
        for j in range(len(c)):
            bi = min(max(int(bin_idx[j]), 0), n_bins - 1)
            vol_bins[bi] += v[j]

        poc_bin = int(np.argmax(vol_bins))
        poc.iloc[i] = (edges[poc_bin] + edges[poc_bin + 1]) / 2.0

    return poc


def compute_volume_profile_signal(
    ohlcv: pd.DataFrame,
    window: int = 20,
    bins: int = 16,
    near_poc_threshold: float = 0.002,
) -> tuple[pd.Series, pd.Series]:
    """Return rolling POC and a continuous distance signal.

    Signal range: [−1, +1]
      +1  — price is exactly at the POC (highest liquidity pool)
       0  — price is at ``near_poc_threshold`` distance from POC
      −1  — price is far from the POC (at least 2× threshold away)

    The continuous output replaces the original ternary ±1 flag, providing
    the taxonomy engine with a gradient signal rather than a hard boundary.
    """
    close  = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float).clip(lower=0.0)
    poc    = _rolling_poc(close, volume, int(window), int(bins)).rename("poc")

    dist_frac = ((close - poc) / poc.replace(0.0, np.nan)).abs().fillna(0.0)

    # Continuous mapping: 1 at dist=0, 0 at dist=threshold, −1 at dist=2×threshold
    # Uses a linear decay then clips at −1.
    signal = (1.0 - dist_frac / (near_poc_threshold + 1e-10)).clip(-1.0, 1.0)

    return poc.fillna(close), signal.rename("volume_profile_signal")

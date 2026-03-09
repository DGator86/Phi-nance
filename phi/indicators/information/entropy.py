"""Rolling Shannon entropy indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-12


def _entropy_from_hist(values: np.ndarray, bins: int, base: float) -> float:
    """Compute normalized entropy in [-1, 1] from a 1D histogram sample."""
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist.astype(float) / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0

    entropy = -np.sum(probs * (np.log(probs + _EPS) / np.log(base)))
    max_entropy = np.log(bins) / np.log(base)
    if max_entropy <= 0:
        return 0.0
    normalized = np.clip(entropy / max_entropy, 0.0, 1.0)
    return float(2.0 * (normalized - 0.5))


def compute_entropy_signal(
    ohlcv: pd.DataFrame,
    window: int = 20,
    bins: int = 20,
    base: float = 2.0,
) -> pd.Series:
    """Compute rolling Shannon entropy of returns, normalized to [-1, 1]."""
    close = ohlcv.get("close")
    if close is None:
        logger.warning("Entropy indicator requires 'close' column; returning zeros.")
        return pd.Series(0.0, index=ohlcv.index)

    if window < 2:
        logger.warning("Entropy window too short (%s); using 2.", window)
        window = 2

    returns = close.astype(float).pct_change()
    if returns.isna().all():
        logger.warning("Entropy indicator received all-NaN returns; returning zeros.")
        return pd.Series(0.0, index=ohlcv.index)

    out = pd.Series(np.nan, index=ohlcv.index, dtype=float)
    valid_count = returns.notna().sum()
    if valid_count < window:
        logger.warning("Entropy indicator needs at least %s return points (got %s).", window, valid_count)

    for i in range(window, len(returns) + 1):
        window_slice = returns.iloc[i - window : i].dropna().values
        if window_slice.size < max(2, window // 2):
            continue
        out.iloc[i - 1] = _entropy_from_hist(window_slice, bins=bins, base=base)

    return out.fillna(0.0).clip(-1.0, 1.0)

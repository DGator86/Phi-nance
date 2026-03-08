"""Rolling Fisher information proxy indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.logging import get_logger

logger = get_logger(__name__)


def compute_fisher_information_signal(
    ohlcv: pd.DataFrame,
    window: int = 20,
    clip_percentile: float = 95.0,
) -> pd.Series:
    """Compute Fisher information proxy (inverse variance) normalized to [-1, 1]."""
    close = ohlcv.get("close")
    if close is None:
        logger.warning("Fisher information requires 'close' column; returning zeros.")
        return pd.Series(0.0, index=ohlcv.index)

    if window < 2:
        logger.warning("Fisher information window too short (%s); using 2.", window)
        window = 2

    returns = close.astype(float).pct_change()
    variance = returns.rolling(window=window, min_periods=max(2, window // 2)).var()
    fisher_proxy = 1.0 / (variance + 1e-10)

    finite = fisher_proxy.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(0.0, index=ohlcv.index)

    upper = np.percentile(finite.values, np.clip(clip_percentile, 50.0, 99.9))
    clipped = fisher_proxy.clip(lower=0.0, upper=max(upper, 1e-10))
    normalized = clipped / max(upper, 1e-10)
    return (2.0 * (normalized - 0.5)).fillna(0.0).clip(-1.0, 1.0)

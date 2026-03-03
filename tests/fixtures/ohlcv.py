"""
tests.fixtures.ohlcv
=====================
Shared OHLCV fixture factories for the test suite.
"""

from __future__ import annotations

import pandas as pd


def make_ohlcv(n: int = 50, start: str = "2023-01-01", negative: bool = False) -> pd.DataFrame:
    """Build a minimal synthetic OHLCV DataFrame.

    Parameters
    ----------
    n        : int — number of daily bars
    start    : str — start date
    negative : bool — insert a negative close at row 2 for sanity-check tests

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    """
    import numpy as np

    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.standard_normal(n))
    if negative:
        closes[2] = -1.0

    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": rng.integers(100_000, 1_000_000, size=n).astype(float),
        },
        index=idx,
    )

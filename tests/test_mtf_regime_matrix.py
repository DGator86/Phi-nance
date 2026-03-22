"""phi.regime.mtf_matrix — resampled timeframe regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.regime.mtf_matrix import (
    build_regime_matrix,
    confluence_score,
    filter_compatible_timeframe_rules,
    resample_ohlcv,
)


def _daily_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 0.4, size=n))
    high = close + rng.uniform(0, 0.5, size=n)
    low = close - rng.uniform(0, 0.5, size=n)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.1, size=n)
    vol = rng.integers(1_000_000, 5_000_000, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_filter_compatible_skips_subdaily_for_daily_bars() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    kept, skip = filter_compatible_timeframe_rules(idx, ("1min", "1H", "1D", "1W"))
    assert "1min" in skip
    assert "1H" in skip
    assert "1D" in kept
    assert "1W" in kept


def test_resample_ohlcv_weekly() -> None:
    o = _daily_ohlcv(80)
    w = resample_ohlcv(o, "1W")
    assert len(w) < len(o)
    assert {"open", "high", "low", "close"}.issubset(w.columns)


def test_build_regime_matrix_daily_has_columns() -> None:
    o = _daily_ohlcv(220)
    mat, meta = build_regime_matrix(o, min_resampled_bars=30)
    assert mat.shape[1] >= 1
    assert "per_timeframe" in meta
    c = confluence_score(mat)
    assert len(c) == len(o)
    assert -1.05 <= float(c.iloc[-1]) <= 1.05

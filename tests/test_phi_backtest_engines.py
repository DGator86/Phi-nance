from __future__ import annotations

import pandas as pd
import pytest

from phi.backtest import get_engine
from phi.run_config import RunConfig


def _ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    close = pd.Series(range(100, 140), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )


def test_vectorized_engine_runs() -> None:
    cfg = RunConfig(indicators={"RSI": {"params": {}}}, blend_method="weighted_sum")
    out = get_engine("vectorized").run(cfg, _ohlcv())
    assert "metrics" in out
    assert "total_return" in out["metrics"]


def test_unknown_engine_raises() -> None:
    with pytest.raises(ValueError):
        get_engine("nope")

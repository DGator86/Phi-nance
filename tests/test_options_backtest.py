"""Unit tests for phi/options/backtest.py."""

from __future__ import annotations

from datetime import date

import pandas as pd

from phi.options.backtest import compute_greeks, run_options_backtest
from phi.run_config import RunConfig


def _make_ohlcv(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [1000] * n},
        index=idx,
    )


def test_run_options_backtest_basic():
    ohlcv = _make_ohlcv(20)
    cfg = RunConfig(
        symbols=["SPY"],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 20),
        trading_mode="options",
        option_params={
            "SPY": {"option_type": "call", "strike": 100.0, "expiry": date(2023, 1, 15), "iv": 0.3, "r": 0.02}
        },
    )
    result = run_options_backtest(cfg, ohlcv)
    assert "portfolio_value" in result
    assert "total_return" in result
    assert "cagr" in result
    assert "max_drawdown" in result
    assert "sharpe" in result
    assert "trades" in result
    assert len(result["portfolio_value"]) == 20


def test_run_options_backtest_legacy_mode():
    ohlcv = _make_ohlcv(15)
    result = run_options_backtest(ohlcv, symbol="SPY", strategy_type="long_put")
    assert isinstance(result["total_return"], float)


def test_compute_greeks_basic():
    g = compute_greeks(100.0, 100.0, 0.5, 0.02, 0.3, "call")
    assert 0 <= g["delta"] <= 1
    assert g["gamma"] >= 0
    assert g["vega"] >= 0

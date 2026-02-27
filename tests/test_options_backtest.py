"""Unit tests for phi/options/backtest.py."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from phi.options.backtest import compute_greeks, run_options_backtest


def _make_ohlcv(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [1000] * n},
        index=idx,
    )


@patch("phi.options.backtest.get_marketdataapp_snapshot", return_value=None)
def test_run_options_backtest_basic(mock_snap):
    ohlcv = _make_ohlcv(20)
    result = run_options_backtest(ohlcv, symbol="SPY", strategy_type="long_call")
    assert "portfolio_value" in result
    assert "total_return" in result
    assert "cagr" in result
    assert "max_drawdown" in result
    assert "sharpe" in result
    assert "trades" in result
    assert len(result["portfolio_value"]) == 20


@patch("phi.options.backtest.get_marketdataapp_snapshot", return_value=None)
def test_run_options_backtest_too_short(mock_snap):
    ohlcv = _make_ohlcv(1)
    result = run_options_backtest(ohlcv, symbol="SPY")
    assert result["total_return"] == 0
    assert result["cagr"] == 0
    assert result["max_drawdown"] == 0
    assert result["sharpe"] == 0


@patch("phi.options.backtest.get_marketdataapp_snapshot", return_value=None)
def test_run_options_backtest_long_put(mock_snap):
    ohlcv = _make_ohlcv(15)
    result = run_options_backtest(ohlcv, symbol="SPY", strategy_type="long_put")
    assert "portfolio_value" in result
    assert isinstance(result["total_return"], float)


def test_compute_greeks_basic():
    g = compute_greeks(0.5, gamma=0.05, theta=-0.02, vega=0.10)
    assert g["delta"] == 0.5
    assert g["gamma"] == 0.05
    assert g["theta"] == -0.02
    assert g["vega"] == 0.10


def test_compute_greeks_defaults():
    g = compute_greeks(0.4)
    assert g["delta"] == 0.4
    assert g["gamma"] is None
    assert g["theta"] is None
    assert g["vega"] is None

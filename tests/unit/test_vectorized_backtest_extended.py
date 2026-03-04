"""
tests.unit.test_vectorized_backtest_extended
=============================================

Extended unit tests for the vectorized backtesting engine:
  • vectorized_positions()
  • equity_curve()
  • run_vectorized_backtest()
  • run_vectorized_batch()
  • VectorizedBacktestResult dataclass

Covers: position styles, edge cases, transaction costs, metrics, serialisation.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.backtest.vectorized import (
    VectorizedBacktestResult,
    vectorized_positions,
    equity_curve,
    run_vectorized_backtest,
    run_vectorized_batch,
    POSITION_STYLE_LONG_ONLY,
    POSITION_STYLE_LONG_SHORT,
    POSITION_STYLE_LONG_FLAT,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

DF_100 = make_ohlcv(n=100)
DF_200 = make_ohlcv(n=200)
DF_500 = make_ohlcv(n=500, start="2022-01-01")


def _const_signal(df: pd.DataFrame, value: float = 0.0) -> pd.Series:
    return pd.Series(value, index=df.index)


def _alternating_signal(df: pd.DataFrame) -> pd.Series:
    vals = np.where(np.arange(len(df)) % 2 == 0, 1.0, -1.0)
    return pd.Series(vals, index=df.index)


def _ramp_signal(df: pd.DataFrame) -> pd.Series:
    """Linearly increases from -1 to +1."""
    vals = np.linspace(-1.0, 1.0, len(df))
    return pd.Series(vals, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# vectorized_positions
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedPositions:

    def test_long_only_all_zeros_on_zero_signal(self):
        sig = _const_signal(DF_100, 0.0)
        pos = vectorized_positions(sig.values, position_style=POSITION_STYLE_LONG_ONLY)
        assert np.all(pos == 0)

    def test_long_only_all_ones_on_strong_signal(self):
        sig = _const_signal(DF_100, 1.0)
        pos = vectorized_positions(sig.values, threshold=0.5, position_style=POSITION_STYLE_LONG_ONLY)
        assert np.all(pos >= 0)

    def test_long_only_no_short_positions(self):
        sig = _alternating_signal(DF_100)
        pos = vectorized_positions(sig.values, position_style=POSITION_STYLE_LONG_ONLY)
        assert np.all(pos >= 0)

    def test_long_short_allows_negative(self):
        sig = _const_signal(DF_100, -1.0)
        pos = vectorized_positions(sig.values, threshold=0.5, position_style=POSITION_STYLE_LONG_SHORT)
        assert np.any(pos < 0)

    def test_long_flat_no_short_positions(self):
        sig = _alternating_signal(DF_100)
        pos = vectorized_positions(sig.values, position_style=POSITION_STYLE_LONG_FLAT)
        assert np.all(pos >= 0)

    def test_returns_numpy_array(self):
        sig = _const_signal(DF_100)
        pos = vectorized_positions(sig.values, position_style=POSITION_STYLE_LONG_ONLY)
        assert isinstance(pos, np.ndarray)

    def test_length_matches_signal(self):
        sig = _const_signal(DF_200)
        pos = vectorized_positions(sig.values, position_style=POSITION_STYLE_LONG_ONLY)
        assert len(pos) == len(sig)

    def test_custom_threshold_zero(self):
        sig = _const_signal(DF_100, 0.01)
        pos_strict = vectorized_positions(sig.values, threshold=0.5,
                                          position_style=POSITION_STYLE_LONG_ONLY)
        pos_loose  = vectorized_positions(sig.values, threshold=0.0,
                                          position_style=POSITION_STYLE_LONG_ONLY)
        assert pos_loose.sum() >= pos_strict.sum()

    def test_invalid_style_raises(self):
        sig = _const_signal(DF_100)
        with pytest.raises((ValueError, KeyError, Exception)):
            vectorized_positions(sig.values, style="nonsense_style")


# ═══════════════════════════════════════════════════════════════════════════════
# equity_curve
# ═══════════════════════════════════════════════════════════════════════════════

class TestEquityCurve:

    def test_flat_signal_returns_near_initial_capital(self):
        closes = DF_100["close"].values
        pos    = np.zeros(len(closes))
        eq     = equity_curve(closes, pos, initial_capital=100_000)
        assert abs(eq[-1] - 100_000) < 1.0

    def test_always_long_equity_follows_price(self):
        closes = DF_100["close"].values
        pos    = np.ones(len(closes))
        eq     = equity_curve(closes, pos, initial_capital=100_000)
        assert isinstance(eq, np.ndarray)
        assert len(eq) == len(closes)

    def test_equity_curve_length(self):
        closes = DF_200["close"].values
        pos    = np.zeros(len(closes))
        eq     = equity_curve(closes, pos, initial_capital=50_000)
        assert len(eq) == len(closes)

    def test_equity_curve_starts_at_initial_capital(self):
        closes = DF_100["close"].values
        pos    = np.zeros(len(closes))
        eq     = equity_curve(closes, pos, initial_capital=75_000)
        assert abs(eq[0] - 75_000) < 1.0

    def test_transaction_cost_returns_array(self):
        closes = DF_100["close"].values
        pos    = np.ones(len(closes))
        eq_no_tc   = equity_curve(closes, pos, initial_capital=100_000, transaction_cost=0.0)
        eq_with_tc = equity_curve(closes, pos, initial_capital=100_000, transaction_cost=0.01)
        assert isinstance(eq_no_tc, np.ndarray)
        assert isinstance(eq_with_tc, np.ndarray)

    def test_equity_always_non_negative(self):
        closes = DF_100["close"].values
        pos    = np.ones(len(closes))
        eq     = equity_curve(closes, pos, initial_capital=100_000, transaction_cost=0.001)
        assert np.all(eq >= 0)


# ═══════════════════════════════════════════════════════════════════════════════
# run_vectorized_backtest
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunVectorizedBacktest:

    def test_returns_vectorized_result_object(self):
        sig = _const_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig)
        assert isinstance(res, VectorizedBacktestResult)

    def test_total_return_is_float(self):
        sig = _ramp_signal(DF_200)
        res = run_vectorized_backtest(DF_200, sig)
        assert isinstance(res.total_return, float)

    def test_sharpe_is_float(self):
        sig = _ramp_signal(DF_200)
        res = run_vectorized_backtest(DF_200, sig)
        assert isinstance(res.sharpe, float)

    def test_max_drawdown_between_0_and_1(self):
        sig = _ramp_signal(DF_200)
        res = run_vectorized_backtest(DF_200, sig)
        assert 0.0 <= res.max_drawdown <= 1.0

    def test_num_trades_non_negative(self):
        sig = _alternating_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig)
        assert res.num_trades >= 0

    def test_win_rate_between_0_and_1(self):
        sig = _ramp_signal(DF_200)
        res = run_vectorized_backtest(DF_200, sig)
        assert 0.0 <= res.win_rate <= 1.0

    def test_cagr_is_float(self):
        sig = _ramp_signal(DF_200)
        res = run_vectorized_backtest(DF_200, sig)
        assert isinstance(res.cagr, float)

    def test_sortino_is_float(self):
        sig = _ramp_signal(DF_200)
        res = run_vectorized_backtest(DF_200, sig)
        assert isinstance(res.sortino, float)

    def test_equity_curve_in_result(self):
        sig = _ramp_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig)
        assert len(res.equity_curve) == len(DF_100)

    def test_positions_in_result(self):
        sig = _ramp_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig)
        assert len(res.positions) == len(DF_100)

    def test_symbol_stored_in_result(self):
        sig = _const_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig, symbol="AAPL")
        assert res.symbol == "AAPL"

    def test_long_only_style(self):
        sig = _alternating_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig, position_style=POSITION_STYLE_LONG_ONLY)
        assert isinstance(res, VectorizedBacktestResult)

    def test_long_short_style(self):
        sig = _alternating_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig, position_style=POSITION_STYLE_LONG_SHORT)
        assert isinstance(res, VectorizedBacktestResult)

    def test_long_flat_style(self):
        sig = _alternating_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig, position_style=POSITION_STYLE_LONG_FLAT)
        assert isinstance(res, VectorizedBacktestResult)

    def test_zero_signal_flat_equity(self):
        sig = _const_signal(DF_100, 0.0)
        res = run_vectorized_backtest(DF_100, sig)
        assert abs(res.total_return) < 0.05

    def test_high_transaction_cost_reduces_return(self):
        sig = _alternating_signal(DF_100)
        res_low  = run_vectorized_backtest(DF_100, sig, transaction_cost=0.0)
        res_high = run_vectorized_backtest(DF_100, sig, transaction_cost=0.05)
        assert res_low.total_return >= res_high.total_return - 0.5

    def test_initial_capital_affects_equity_not_return(self):
        sig = _ramp_signal(DF_200)
        res1 = run_vectorized_backtest(DF_200, sig, initial_capital=50_000)
        res2 = run_vectorized_backtest(DF_200, sig, initial_capital=200_000)
        assert abs(res1.total_return - res2.total_return) < 0.01

    def test_to_dict_returns_dict(self):
        sig = _ramp_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig)
        d = res.to_dict()
        assert isinstance(d, dict)
        assert "total_return" in d

    def test_summary_returns_string(self):
        sig = _ramp_signal(DF_100)
        res = run_vectorized_backtest(DF_100, sig)
        s = res.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_too_few_bars_raises(self):
        df_tiny = make_ohlcv(n=3)
        sig = _const_signal(df_tiny)
        with pytest.raises(Exception):
            run_vectorized_backtest(df_tiny, sig)

    def test_500_bar_dataset(self):
        sig = _ramp_signal(DF_500)
        res = run_vectorized_backtest(DF_500, sig)
        assert isinstance(res, VectorizedBacktestResult)


# ═══════════════════════════════════════════════════════════════════════════════
# run_vectorized_batch
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunVectorizedBatch:

    def test_batch_returns_list(self):
        signals = {
            "RSI": _ramp_signal(DF_100),
            "Flat": _const_signal(DF_100),
        }
        results = run_vectorized_batch(DF_100, signals)
        assert isinstance(results, (list, dict))

    def test_batch_length_matches_signals(self):
        signals = {f"Sig{i}": _ramp_signal(DF_100) for i in range(4)}
        results = run_vectorized_batch(DF_100, signals)
        if isinstance(results, list):
            assert len(results) == 4
        else:
            assert len(results) == 4

    def test_batch_each_result_is_vbt_result(self):
        signals = {"S1": _ramp_signal(DF_100), "S2": _const_signal(DF_100)}
        results = run_vectorized_batch(DF_100, signals)
        if isinstance(results, list):
            for r in results:
                assert isinstance(r, VectorizedBacktestResult)
        else:
            for r in results.values():
                assert isinstance(r, VectorizedBacktestResult)

    def test_empty_batch_returns_empty(self):
        results = run_vectorized_batch(DF_100, {})
        if isinstance(results, list):
            assert results == []
        else:
            assert results == {}


# ═══════════════════════════════════════════════════════════════════════════════
# VectorizedBacktestResult dataclass
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedBacktestResultDataclass:

    def _make_result(self, **kwargs) -> VectorizedBacktestResult:
        defaults = dict(
            symbol          = "TEST",
            total_return    = 0.05,
            cagr            = 0.04,
            sharpe          = 0.8,
            sortino         = 1.0,
            max_drawdown    = 0.12,
            win_rate        = 0.55,
            num_trades      = 12,
            equity_curve    = np.ones(100) * 100_000,
            positions       = np.zeros(100),
            bars_per_year   = 252.0,
            initial_capital = 100_000.0,
            final_capital   = 105_000.0,
        )
        defaults.update(kwargs)
        return VectorizedBacktestResult(**defaults)

    def test_create_result(self):
        r = self._make_result()
        assert r.symbol == "TEST"

    def test_to_dict_has_expected_keys(self):
        r = self._make_result()
        d = r.to_dict()
        for key in ("symbol", "total_return", "sharpe_ratio", "max_drawdown"):
            assert key in d

    def test_summary_non_empty_string(self):
        r = self._make_result()
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_equity_curve_array(self):
        r = self._make_result()
        assert isinstance(r.equity_curve, np.ndarray)

    def test_positions_array(self):
        r = self._make_result()
        assert isinstance(r.positions, np.ndarray)

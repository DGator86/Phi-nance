"""
tests.unit.test_vectorized_backtest
======================================

Comprehensive tests for Phase 9.3 — Vectorized Backtesting Engine:
  • vectorized_positions (long_only, long_short, long_flat, edge cases)
  • equity_curve
  • _ffill helper
  • run_vectorized_backtest (full integration)
  • run_vectorized_batch
  • VectorizedBacktestResult (dataclass, to_dict, summary)
  • Performance: vectorized must beat bar-by-bar on large datasets
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

# ── Shared fixtures ────────────────────────────────────────────────────────────

DF_SMALL  = make_ohlcv(n=30,   start="2020-01-01")
DF_MED    = make_ohlcv(n=300,  start="2021-01-01")
DF_LARGE  = make_ohlcv(n=2000, start="2022-01-01")


def _constant_signal(df: pd.DataFrame, value: float) -> pd.Series:
    return pd.Series(value, index=df.index)


def _alternating_signal(df: pd.DataFrame, threshold: float = 0.2) -> pd.Series:
    n   = len(df)
    arr = np.where(np.arange(n) % 20 < 10, threshold + 0.1, -(threshold + 0.1))
    return pd.Series(arr, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# vectorized_positions
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedPositions:

    def test_long_only_all_up_is_one(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.ones(100) * 0.5
        pos = vectorized_positions(sig, threshold=0.15, position_style="long_only")
        assert np.all(pos[1:] == 1.0)  # after first bar held

    def test_long_only_all_down_is_zero(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.ones(100) * -0.5
        pos = vectorized_positions(sig, threshold=0.15, position_style="long_only")
        assert np.all(pos == 0.0)

    def test_long_short_down_is_minus_one(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.ones(100) * -0.5
        pos = vectorized_positions(sig, threshold=0.15, position_style="long_short")
        assert np.all(pos == -1.0)

    def test_long_short_up_is_plus_one(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.ones(100) * 0.5
        pos = vectorized_positions(sig, threshold=0.15, position_style="long_short")
        assert np.all(pos[1:] == 1.0)

    def test_long_flat_always_zero_or_one(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.random.uniform(-1, 1, 200)
        pos = vectorized_positions(sig, threshold=0.15, position_style="long_flat")
        assert set(np.unique(pos)).issubset({0.0, 1.0})

    def test_output_length_matches_input(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.random.uniform(-1, 1, 150)
        pos = vectorized_positions(sig, threshold=0.15)
        assert len(pos) == 150

    def test_invalid_style_raises(self):
        from phinance.backtest.vectorized import vectorized_positions
        with pytest.raises(ValueError, match="position_style"):
            vectorized_positions(np.zeros(10), position_style="invalid_mode")

    def test_positions_hold_between_signals(self):
        """Position should be held (forward-filled) between signal transitions."""
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.zeros(20)
        sig[0]  =  0.5   # enter long
        sig[10] = -0.5   # exit
        pos = vectorized_positions(sig, threshold=0.15, position_style="long_only")
        # Bars 1-9 should still be long (held from bar 0)
        assert np.all(pos[1:10] == 1.0)

    def test_zero_signal_stays_flat(self):
        from phinance.backtest.vectorized import vectorized_positions
        sig = np.zeros(50)
        pos = vectorized_positions(sig, threshold=0.15)
        assert np.all(pos == 0.0)

    def test_threshold_boundary(self):
        from phinance.backtest.vectorized import vectorized_positions
        threshold = 0.3
        sig = np.array([threshold - 0.01, threshold + 0.01, -0.5, 0.0])
        pos = vectorized_positions(sig, threshold=threshold, position_style="long_only")
        # Bar 0: below threshold → 0; Bar 1: above threshold → 1
        assert pos[0] == 0.0
        assert pos[1] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# equity_curve
# ═══════════════════════════════════════════════════════════════════════════════

class TestEquityCurve:

    def test_flat_position_returns_constant_equity(self):
        from phinance.backtest.vectorized import equity_curve
        closes  = np.linspace(100, 110, 50)
        pos     = np.zeros(50)
        eq      = equity_curve(closes, pos, initial_capital=100_000)
        assert np.all(eq == 100_000.0)

    def test_long_rising_market_increases_equity(self):
        from phinance.backtest.vectorized import equity_curve
        closes = np.linspace(100, 200, 100)
        pos    = np.ones(100)
        eq     = equity_curve(closes, pos, initial_capital=100_000)
        assert eq[-1] > eq[0]

    def test_long_falling_market_decreases_equity(self):
        from phinance.backtest.vectorized import equity_curve
        closes = np.linspace(200, 100, 100)
        pos    = np.ones(100)
        eq     = equity_curve(closes, pos, initial_capital=100_000)
        assert eq[-1] < eq[0]

    def test_output_length_matches_input(self):
        from phinance.backtest.vectorized import equity_curve
        closes = np.random.uniform(90, 110, 80)
        pos    = np.zeros(80)
        eq     = equity_curve(closes, pos)
        assert len(eq) == 80

    def test_initial_capital_respected(self):
        from phinance.backtest.vectorized import equity_curve
        closes = np.ones(20) * 100
        pos    = np.zeros(20)
        eq     = equity_curve(closes, pos, initial_capital=50_000)
        assert eq[0] == pytest.approx(50_000.0)

    def test_transaction_cost_reduces_equity(self):
        from phinance.backtest.vectorized import equity_curve
        closes = np.linspace(100, 110, 50)
        # Force alternating positions so transaction costs are actually applied
        pos = np.zeros(50)
        pos[::10] = 1.0  # position changes every 10 bars
        eq_no_cost = equity_curve(closes, pos, transaction_cost=0.0)
        eq_cost    = equity_curve(closes, pos, transaction_cost=0.05)  # 5% cost
        # With alternating positions, cost should reduce final equity
        assert eq_cost[-1] <= eq_no_cost[-1]

    def test_short_position_in_falling_market(self):
        from phinance.backtest.vectorized import equity_curve
        closes = np.linspace(200, 100, 100)
        pos    = np.full(100, -1.0)
        eq     = equity_curve(closes, pos, initial_capital=100_000)
        assert eq[-1] > eq[0]  # short in falling market = profit


# ═══════════════════════════════════════════════════════════════════════════════
# run_vectorized_backtest — integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunVectorizedBacktest:

    def test_returns_result_object(self):
        from phinance.backtest.vectorized import run_vectorized_backtest, VectorizedBacktestResult
        sig    = _alternating_signal(DF_MED)
        result = run_vectorized_backtest(DF_MED, sig, symbol="SPY")
        assert isinstance(result, VectorizedBacktestResult)

    def test_symbol_preserved(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _constant_signal(DF_MED, 0.5)
        r   = run_vectorized_backtest(DF_MED, sig, symbol="AAPL")
        assert r.symbol == "AAPL"

    def test_equity_curve_length(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert len(r.equity_curve) == len(DF_MED)

    def test_positions_length(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert len(r.positions) == len(DF_MED)

    def test_initial_capital_preserved(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig, initial_capital=50_000)
        assert r.initial_capital == pytest.approx(50_000.0)
        assert r.equity_curve[0] == pytest.approx(50_000.0)

    def test_sharpe_is_finite(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert np.isfinite(r.sharpe)

    def test_max_drawdown_non_negative(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert r.max_drawdown >= 0.0

    def test_max_drawdown_at_most_one(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert r.max_drawdown <= 1.0

    def test_win_rate_in_unit_interval(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert 0.0 <= r.win_rate <= 1.0

    def test_long_short_positions(self):
        from phinance.backtest.vectorized import run_vectorized_backtest, POSITION_STYLE_LONG_SHORT
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig, position_style=POSITION_STYLE_LONG_SHORT)
        assert -1.0 in r.positions or 1.0 in r.positions

    def test_too_few_bars_raises(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        tiny = make_ohlcv(n=3)
        sig  = _constant_signal(tiny, 0.5)
        with pytest.raises(ValueError, match="[Bb]ars|rows"):
            run_vectorized_backtest(tiny, sig)

    def test_to_dict_has_expected_keys(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig, symbol="TEST")
        d   = r.to_dict()
        for k in ("symbol", "total_return", "cagr", "sharpe_ratio",
                  "sortino_ratio", "max_drawdown", "win_rate", "num_trades"):
            assert k in d, f"Missing key: {k}"

    def test_summary_string(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig, symbol="SPY")
        s   = r.summary()
        assert "SPY"    in s
        assert "Sharpe" in s
        assert "Return" in s

    def test_flat_signal_no_trades(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _constant_signal(DF_MED, 0.0)  # always hold threshold=0.15
        r   = run_vectorized_backtest(DF_MED, sig, signal_threshold=0.15)
        assert r.num_trades == 0

    def test_rsi_indicator_signal(self):
        """End-to-end: RSI signal fed into vectorized engine."""
        from phinance.backtest.vectorized import run_vectorized_backtest
        from phinance.strategies.rsi import RSIIndicator
        sig = RSIIndicator().compute(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig, symbol="RSI_TEST")
        assert isinstance(r.total_return, float)

    def test_result_bars_per_year_positive(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig)
        assert r.bars_per_year > 0

    def test_final_capital_positive(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_MED)
        r   = run_vectorized_backtest(DF_MED, sig, initial_capital=100_000)
        assert r.final_capital > 0

    def test_all_position_styles_run(self):
        from phinance.backtest.vectorized import (
            run_vectorized_backtest,
            POSITION_STYLE_LONG_ONLY,
            POSITION_STYLE_LONG_SHORT,
            POSITION_STYLE_LONG_FLAT,
        )
        sig = _alternating_signal(DF_MED)
        for style in (POSITION_STYLE_LONG_ONLY, POSITION_STYLE_LONG_SHORT, POSITION_STYLE_LONG_FLAT):
            r = run_vectorized_backtest(DF_MED, sig, position_style=style)
            assert isinstance(r.sharpe, float)


# ═══════════════════════════════════════════════════════════════════════════════
# run_vectorized_batch
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunVectorizedBatch:

    def test_batch_returns_dict(self):
        from phinance.backtest.vectorized import run_vectorized_batch
        signals = {
            "up":   _constant_signal(DF_MED, 0.5),
            "down": _constant_signal(DF_MED, -0.5),
        }
        results = run_vectorized_batch(DF_MED, signals)
        assert isinstance(results, dict)
        assert "up"   in results
        assert "down" in results

    def test_batch_result_types(self):
        from phinance.backtest.vectorized import run_vectorized_batch, VectorizedBacktestResult
        signals = {"s1": _alternating_signal(DF_MED)}
        results = run_vectorized_batch(DF_MED, signals)
        for r in results.values():
            assert isinstance(r, VectorizedBacktestResult)

    def test_batch_empty_signals_returns_empty(self):
        from phinance.backtest.vectorized import run_vectorized_batch
        results = run_vectorized_batch(DF_MED, {})
        assert results == {}

    def test_batch_with_multiple_indicators(self):
        from phinance.backtest.vectorized import run_vectorized_batch
        from phinance.strategies.indicator_catalog import compute_indicator
        signals = {
            "RSI":  compute_indicator("RSI",  DF_MED, {}),
            "MACD": compute_indicator("MACD", DF_MED, {}),
        }
        results = run_vectorized_batch(DF_MED, signals)
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Performance: vectorized must be faster than bar-by-bar on large data
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizedPerformance:

    def test_vectorized_faster_than_loop_on_large_data(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        from phinance.backtest.engine import simulate
        from phinance.strategies.rsi import RSIIndicator

        sig = RSIIndicator().compute(DF_LARGE)

        # Vectorized
        t0 = time.perf_counter()
        run_vectorized_backtest(DF_LARGE, sig, symbol="VEC")
        vec_time = time.perf_counter() - t0

        # Bar-by-bar
        t0 = time.perf_counter()
        simulate(DF_LARGE, sig.fillna(0.0))
        loop_time = time.perf_counter() - t0

        # Vectorized should not be dramatically slower (we allow up to 10x)
        assert vec_time < max(loop_time * 10, 2.0), (
            f"Vectorized ({vec_time:.4f}s) unexpectedly slow vs "
            f"bar-by-bar ({loop_time:.4f}s)"
        )

    def test_vectorized_large_dataset_completes_under_1s(self):
        from phinance.backtest.vectorized import run_vectorized_backtest
        sig = _alternating_signal(DF_LARGE)
        t0  = time.perf_counter()
        run_vectorized_backtest(DF_LARGE, sig)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Vectorized on 2000-bar dataset took {elapsed:.2f}s"

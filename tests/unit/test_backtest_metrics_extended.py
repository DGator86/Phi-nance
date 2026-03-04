"""
tests.unit.test_backtest_metrics_extended
==========================================

Extended unit tests for phinance.backtest.metrics and .models:
  • bars_per_year(df)
  • total_return(pv, initial_capital)
  • cagr(pv, initial_capital, bpy)
  • max_drawdown(pv)
  • sharpe_ratio(pv, bpy)
  • sortino_ratio(pv, bpy)
  • win_rate(trades)
  • compute_all(portfolio_values, ohlcv, initial_capital, trades=None)
  • Trade dataclass — win property
"""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.backtest.metrics import (
    bars_per_year,
    total_return,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
    compute_all,
)
from phinance.backtest.models import Trade

# ── Helpers ───────────────────────────────────────────────────────────────────

DF_252 = make_ohlcv(n=252)
DF_504 = make_ohlcv(n=504, start="2022-01-01")


def _flat_equity(n=252, value=100_000.0) -> np.ndarray:
    return np.full(n, value)


def _growing_equity(n=252, start=100_000.0, rate=0.1) -> np.ndarray:
    return start * (1 + rate) ** (np.arange(n) / n)


def _declining_equity(n=252, start=100_000.0, rate=-0.2) -> np.ndarray:
    return start * (1 + rate) ** (np.arange(n) / n)


def _volatile_equity(n=252) -> np.ndarray:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0003, 0.01, n)
    return 100_000 * np.cumprod(1 + returns)


def _make_trade(entry=100.0, exit_=110.0, qty=10, pnl=None, pnl_pct=None) -> Trade:
    if pnl is None:
        pnl = (exit_ - entry) * qty
    if pnl_pct is None:
        pnl_pct = (exit_ - entry) / entry
    return Trade(
        entry_date  = date(2023, 1, 1),
        exit_date   = date(2023, 1, 5),
        symbol      = "SPY",
        entry_price = entry,
        exit_price  = exit_,
        quantity    = qty,
        pnl         = pnl,
        pnl_pct     = pnl_pct,
        hold_bars   = 4,
        direction   = "long",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# bars_per_year
# ═══════════════════════════════════════════════════════════════════════════════

class TestBarsPerYear:

    def test_daily_252(self):
        assert bars_per_year(DF_252) == pytest.approx(252, abs=5)

    def test_larger_dataset_same_scale(self):
        bpy = bars_per_year(DF_504)
        assert bpy == pytest.approx(252, abs=5)

    def test_returns_float(self):
        assert isinstance(bars_per_year(DF_252), float)

    def test_positive(self):
        assert bars_per_year(DF_252) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# total_return
# ═══════════════════════════════════════════════════════════════════════════════

class TestTotalReturn:

    def test_flat_return_is_zero(self):
        eq = _flat_equity()
        assert abs(total_return(eq, 100_000.0)) < 1e-9

    def test_growing_return_positive(self):
        eq = _growing_equity(rate=0.2)
        assert total_return(eq, 100_000.0) > 0

    def test_declining_return_negative(self):
        eq = _declining_equity(rate=-0.3)
        assert total_return(eq, 100_000.0) < 0

    def test_return_100pct(self):
        eq = np.array([100_000.0, 200_000.0])
        assert abs(total_return(eq, 100_000.0) - 1.0) < 1e-9

    def test_return_minus_50pct(self):
        eq = np.array([100_000.0, 50_000.0])
        assert abs(total_return(eq, 100_000.0) - (-0.5)) < 1e-9

    def test_returns_float(self):
        eq = _volatile_equity()
        assert isinstance(total_return(eq, 100_000.0), float)


# ═══════════════════════════════════════════════════════════════════════════════
# cagr
# ═══════════════════════════════════════════════════════════════════════════════

class TestCagr:

    def test_flat_cagr_is_zero(self):
        eq = _flat_equity(252)
        assert abs(cagr(eq, 100_000.0, 252.0)) < 1e-6

    def test_positive_cagr(self):
        eq = _growing_equity(n=504, rate=0.2)  # 2 years
        c = cagr(eq, 100_000.0, 252.0)
        assert c > 0

    def test_negative_cagr(self):
        eq = _declining_equity(n=252, rate=-0.3)
        c = cagr(eq, 100_000.0, 252.0)
        assert c < 0

    def test_returns_float(self):
        eq = _volatile_equity()
        c = cagr(eq, 100_000.0, 252.0)
        assert isinstance(c, float)

    def test_cagr_clipped_for_near_zero_equity(self):
        eq = np.array([100_000.0, 0.01])
        c = cagr(eq, 100_000.0, 252.0)
        assert isinstance(c, float)   # should not raise

    def test_single_bar_no_crash(self):
        eq = np.array([100_000.0])
        c = cagr(eq, 100_000.0, 252.0)
        assert isinstance(c, float)


# ═══════════════════════════════════════════════════════════════════════════════
# max_drawdown
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxDrawdown:

    def test_flat_drawdown_is_zero(self):
        eq = _flat_equity()
        assert abs(max_drawdown(eq)) < 1e-9

    def test_monotone_growing_drawdown_zero(self):
        eq = np.linspace(100_000, 200_000, 100)
        assert max_drawdown(eq) < 1e-9

    def test_drawdown_always_non_negative(self):
        eq = _volatile_equity()
        assert max_drawdown(eq) >= 0

    def test_drawdown_at_most_one(self):
        eq = _volatile_equity()
        assert max_drawdown(eq) <= 1.0

    def test_complete_loss_drawdown_near_one(self):
        eq = np.array([100_000.0, 50_000.0, 0.01])
        dd = max_drawdown(eq)
        assert dd > 0.99

    def test_partial_drawdown(self):
        eq = np.array([100.0, 90.0, 95.0])
        dd = max_drawdown(eq)
        assert abs(dd - 0.10) < 0.01

    def test_returns_float(self):
        eq = _volatile_equity()
        assert isinstance(max_drawdown(eq), float)


# ═══════════════════════════════════════════════════════════════════════════════
# sharpe_ratio
# ═══════════════════════════════════════════════════════════════════════════════

class TestSharpeRatio:

    def test_flat_equity_sharpe_zero(self):
        eq = _flat_equity()
        s = sharpe_ratio(eq, bpy=252.0)
        assert abs(s) < 1e-6

    def test_positive_trend_sharpe_positive(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.005, 252)
        eq = 100_000 * np.cumprod(1 + returns)
        s = sharpe_ratio(eq, bpy=252.0)
        assert s > 0

    def test_sharpe_is_float(self):
        eq = _volatile_equity()
        assert isinstance(sharpe_ratio(eq, bpy=252.0), float)

    def test_sharpe_default_bpy(self):
        eq = _volatile_equity()
        s = sharpe_ratio(eq)
        assert isinstance(s, float)


# ═══════════════════════════════════════════════════════════════════════════════
# sortino_ratio
# ═══════════════════════════════════════════════════════════════════════════════

class TestSortinoRatio:

    def test_flat_equity_sortino_zero(self):
        eq = _flat_equity()
        s = sortino_ratio(eq, bpy=252.0)
        assert abs(s) < 1e-6

    def test_sortino_is_float(self):
        eq = _volatile_equity()
        assert isinstance(sortino_ratio(eq, bpy=252.0), float)

    def test_sortino_default_bpy(self):
        eq = _volatile_equity()
        s = sortino_ratio(eq)
        assert isinstance(s, float)

    def test_positive_returns_positive_sortino(self):
        rng = np.random.default_rng(1)
        returns = abs(rng.normal(0.001, 0.005, 252))
        eq = 100_000 * np.cumprod(1 + returns)
        s = sortino_ratio(eq, bpy=252.0)
        assert s > 0


# ═══════════════════════════════════════════════════════════════════════════════
# win_rate
# ═══════════════════════════════════════════════════════════════════════════════

class TestWinRate:

    def test_all_winning_trades(self):
        trades = [_make_trade(entry=100, exit_=110) for _ in range(5)]
        assert abs(win_rate(trades) - 1.0) < 1e-9

    def test_all_losing_trades(self):
        trades = [_make_trade(entry=110, exit_=100, pnl=-100, pnl_pct=-0.09) for _ in range(5)]
        assert abs(win_rate(trades) - 0.0) < 1e-9

    def test_empty_trades_win_rate_zero(self):
        assert win_rate([]) == 0.0

    def test_mixed_trades(self):
        trades = [
            _make_trade(entry=100, exit_=110),
            _make_trade(entry=110, exit_=100, pnl=-100, pnl_pct=-0.09),
        ]
        assert abs(win_rate(trades) - 0.5) < 1e-9

    def test_single_win(self):
        trades = [_make_trade(entry=100, exit_=110)]
        assert abs(win_rate(trades) - 1.0) < 1e-9

    def test_single_loss(self):
        trades = [_make_trade(entry=110, exit_=100, pnl=-100, pnl_pct=-0.09)]
        assert abs(win_rate(trades) - 0.0) < 1e-9

    def test_win_rate_between_0_and_1(self):
        trades = [_make_trade(entry=100, exit_=110) for _ in range(7)] + \
                 [_make_trade(entry=110, exit_=100, pnl=-100, pnl_pct=-0.09) for _ in range(3)]
        wr = win_rate(trades)
        assert 0.0 <= wr <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# compute_all
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeAll:

    def test_returns_dict(self):
        eq = _volatile_equity().tolist()
        result = compute_all(eq, DF_252, 100_000.0)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        eq = _volatile_equity().tolist()
        result = compute_all(eq, DF_252, 100_000.0)
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "sortino", "win_rate"):
            assert key in result, f"Missing key: {key}"

    def test_values_are_numeric(self):
        eq = _volatile_equity().tolist()
        result = compute_all(eq, DF_252, 100_000.0)
        for k, v in result.items():
            assert isinstance(v, (int, float)), f"{k}={v} is not numeric"

    def test_with_trades(self):
        eq = _volatile_equity().tolist()
        trades = [_make_trade(entry=100, exit_=110)]
        result = compute_all(eq, DF_252, 100_000.0, trades=trades)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Trade model
# ═══════════════════════════════════════════════════════════════════════════════

class TestTradeModel:

    def test_win_on_long_profit(self):
        t = _make_trade(entry=100, exit_=110, pnl=100, pnl_pct=0.1)
        assert t.win is True

    def test_loss_on_long_loss(self):
        t = _make_trade(entry=110, exit_=100, pnl=-100, pnl_pct=-0.09)
        assert t.win is False

    def test_breakeven_not_win(self):
        t = _make_trade(entry=100, exit_=100, pnl=0.0, pnl_pct=0.0)
        assert t.win is False

    def test_trade_symbol(self):
        t = _make_trade()
        assert t.symbol == "SPY"

    def test_trade_pnl(self):
        t = _make_trade(entry=100, exit_=110, pnl=100.0)
        assert t.pnl == 100.0

    def test_trade_repr_or_str(self):
        t = _make_trade()
        s = str(t)
        assert isinstance(s, str)

    def test_trade_direction(self):
        t = _make_trade()
        assert t.direction == "long"

    def test_trade_quantity(self):
        t = _make_trade(qty=25)
        assert t.quantity == 25

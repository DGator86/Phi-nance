"""
tests/unit/test_portfolio_backtest.py
=======================================

Comprehensive unit tests for phinance.backtest.portfolio.

Covers:
  - AllocationMethod enum
  - PortfolioConfig (defaults, custom, to_dict)
  - AssetResult dataclass (creation, to_dict, repr)
  - PortfolioResult dataclass (creation, to_dict, summary, repr)
  - PortfolioBacktester.__init__
  - PortfolioBacktester._compute_allocations (equal, risk_parity, fixed)
  - PortfolioBacktester._simulate_asset (equity curve length, non-negative)
  - PortfolioBacktester._aggregate_equity
  - PortfolioBacktester._correlation_matrix
  - PortfolioBacktester.run (full result, metrics ranges)
  - run_portfolio_backtest convenience function
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phinance.backtest.portfolio import (
    AllocationMethod,
    PortfolioConfig,
    AssetResult,
    PortfolioResult,
    PortfolioBacktester,
    run_portfolio_backtest,
)
from phinance.strategies.indicator_catalog import compute_indicator


# ── fixtures ──────────────────────────────────────────────────────────────────


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.normal(0, 0.002, n)),
            "high":   close * (1 + abs(rng.normal(0, 0.005, n))),
            "low":    close * (1 - abs(rng.normal(0, 0.005, n))),
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


DF_A = make_ohlcv(200, seed=1)
DF_B = make_ohlcv(200, seed=2)
DF_C = make_ohlcv(200, seed=3)


def _signals(ohlcv_dict, name="EMA Cross"):
    return {sym: compute_indicator(name, df, {}) for sym, df in ohlcv_dict.items()}


# ── AllocationMethod ──────────────────────────────────────────────────────────


class TestAllocationMethod:
    def test_equal_value(self):
        assert AllocationMethod.EQUAL.value == "equal"

    def test_risk_parity_value(self):
        assert AllocationMethod.RISK_PARITY.value == "risk_parity"

    def test_fixed_value(self):
        assert AllocationMethod.FIXED.value == "fixed"

    def test_from_string(self):
        assert AllocationMethod("equal") == AllocationMethod.EQUAL


# ── PortfolioConfig ───────────────────────────────────────────────────────────


class TestPortfolioConfig:
    def test_defaults(self):
        cfg = PortfolioConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.allocation == AllocationMethod.EQUAL
        assert cfg.position_size == 1.0
        assert cfg.transaction_cost == 0.001
        assert not cfg.allow_short

    def test_custom(self):
        cfg = PortfolioConfig(
            initial_capital=50_000,
            allocation=AllocationMethod.RISK_PARITY,
            allow_short=True,
        )
        assert cfg.initial_capital == 50_000
        assert cfg.allow_short

    def test_to_dict_keys(self):
        d = PortfolioConfig().to_dict()
        for k in ("initial_capital", "allocation", "position_size",
                  "transaction_cost", "allow_short"):
            assert k in d

    def test_allocation_str_in_dict(self):
        d = PortfolioConfig(allocation=AllocationMethod.FIXED).to_dict()
        assert d["allocation"] == "fixed"


# ── AssetResult ───────────────────────────────────────────────────────────────


class TestAssetResult:
    def test_default_creation(self):
        ar = AssetResult()
        assert ar.symbol == ""
        assert ar.total_return == 0.0
        assert ar.num_trades == 0

    def test_custom_creation(self):
        ar = AssetResult(
            symbol="SPY",
            allocation=0.5,
            total_return=0.12,
            sharpe=1.5,
            num_trades=20,
        )
        assert ar.symbol == "SPY"
        assert ar.sharpe == 1.5

    def test_to_dict_keys(self):
        ar = AssetResult(symbol="QQQ")
        d = ar.to_dict()
        for k in ("symbol", "allocation", "total_return", "sharpe",
                  "max_drawdown", "win_rate", "num_trades"):
            assert k in d

    def test_repr(self):
        ar = AssetResult(symbol="SPY", total_return=0.05, sharpe=1.2)
        r = repr(ar)
        assert "AssetResult" in r
        assert "SPY" in r


# ── PortfolioResult ───────────────────────────────────────────────────────────


class TestPortfolioResult:
    def test_default_creation(self):
        pr = PortfolioResult()
        assert isinstance(pr.portfolio_id, str)
        assert pr.total_return == 0.0
        assert pr.num_trades == 0

    def test_to_dict_keys(self):
        pr = PortfolioResult(symbols=["SPY"])
        d = pr.to_dict()
        for k in ("portfolio_id", "symbols", "initial_capital", "final_capital",
                  "total_return", "cagr", "sharpe", "sortino", "max_drawdown",
                  "win_rate", "num_trades", "asset_results"):
            assert k in d

    def test_summary_string(self):
        pr = PortfolioResult(symbols=["SPY", "QQQ"], total_return=0.15, sharpe=1.2)
        s = pr.summary()
        assert "SPY" in s
        assert "1.200" in s

    def test_repr(self):
        pr = PortfolioResult(symbols=["SPY"])
        assert "PortfolioResult" in repr(pr)

    def test_unique_portfolio_ids(self):
        ids = {PortfolioResult().portfolio_id for _ in range(20)}
        assert len(ids) == 20


# ── PortfolioBacktester.__init__ ──────────────────────────────────────────────


class TestPortfolioBacktesterInit:
    def test_symbols(self):
        ohlcv = {"SPY": DF_A, "QQQ": DF_B}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        assert set(bt.symbols) == {"SPY", "QQQ"}

    def test_default_config(self):
        ohlcv = {"SPY": DF_A}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        assert isinstance(bt.config, PortfolioConfig)

    def test_custom_config(self):
        ohlcv = {"SPY": DF_A}
        cfg = PortfolioConfig(initial_capital=50_000)
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv), config=cfg)
        assert bt.config.initial_capital == 50_000


# ── _compute_allocations ──────────────────────────────────────────────────────


class TestComputeAllocations:
    def _bt(self, ohlcv, allocation, fixed=None):
        cfg = PortfolioConfig(
            allocation=allocation,
            fixed_weights=fixed or {},
        )
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv), config=cfg)
        closes = {s: ohlcv[s]["close"].values for s in bt.symbols}
        signals = {s: np.zeros(len(closes[s])) for s in bt.symbols}
        return bt._compute_allocations(closes, signals)

    def test_equal_sum_to_one(self):
        ohlcv = {"A": DF_A, "B": DF_B, "C": DF_C}
        allocs = self._bt(ohlcv, AllocationMethod.EQUAL)
        assert abs(sum(allocs.values()) - 1.0) < 1e-9

    def test_equal_all_same(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        allocs = self._bt(ohlcv, AllocationMethod.EQUAL)
        assert abs(allocs["A"] - allocs["B"]) < 1e-9

    def test_risk_parity_sum(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        allocs = self._bt(ohlcv, AllocationMethod.RISK_PARITY)
        assert abs(sum(allocs.values()) - 1.0) < 1e-9

    def test_risk_parity_all_positive(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        allocs = self._bt(ohlcv, AllocationMethod.RISK_PARITY)
        assert all(v > 0 for v in allocs.values())

    def test_fixed_weights(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        fixed = {"A": 0.7, "B": 0.3}
        allocs = self._bt(ohlcv, AllocationMethod.FIXED, fixed=fixed)
        assert abs(allocs["A"] - 0.7) < 1e-9
        assert abs(allocs["B"] - 0.3) < 1e-9

    def test_fixed_empty_falls_back_to_equal(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        allocs = self._bt(ohlcv, AllocationMethod.FIXED, fixed={})
        assert abs(sum(allocs.values()) - 1.0) < 1e-9


# ── _simulate_asset ───────────────────────────────────────────────────────────


class TestSimulateAsset:
    def _run_asset(self, df=None, signal=None, capital=10_000):
        df = df or DF_A
        if signal is None:
            signal = compute_indicator("EMA Cross", df, {}).fillna(0.0)
        ohlcv = {"SPY": df}
        bt = PortfolioBacktester(ohlcv, {"SPY": signal})
        closes = df["close"].values.astype(float)
        sig_arr = signal.values.astype(float)
        return bt._simulate_asset("SPY", closes, sig_arr, capital, 1.0)

    def test_returns_tuple(self):
        eq, ar = self._run_asset()
        assert isinstance(eq, np.ndarray)
        assert isinstance(ar, AssetResult)

    def test_equity_length(self):
        eq, _ = self._run_asset()
        assert len(eq) == len(DF_A)

    def test_equity_non_negative(self):
        eq, _ = self._run_asset()
        assert np.all(eq >= 0.0)

    def test_symbol_set(self):
        _, ar = self._run_asset()
        assert ar.symbol == "SPY"

    def test_allocation_set(self):
        _, ar = self._run_asset()
        assert ar.allocation == 1.0

    def test_flat_signal_capital_preserved(self):
        signal = pd.Series(np.zeros(len(DF_A)))
        eq, ar = self._run_asset(signal=signal)
        # Flat → no trades → equity should stay at initial
        assert abs(eq[0] - 10_000) < 1.0


# ── _aggregate_equity ─────────────────────────────────────────────────────────


class TestAggregateEquity:
    def test_length(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        a = np.ones(100) * 5000
        b = np.ones(100) * 5000
        agg = bt._aggregate_equity({"A": a, "B": b}, 100)
        assert len(agg) == 100

    def test_sum_correct(self):
        ohlcv = {"A": DF_A}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        a = np.ones(50) * 5000
        agg = bt._aggregate_equity({"A": a}, 50)
        assert np.allclose(agg, 5000.0)

    def test_two_assets_sum(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        a = np.ones(50) * 3000
        b = np.ones(50) * 7000
        agg = bt._aggregate_equity({"A": a, "B": b}, 50)
        assert np.allclose(agg, 10_000.0)


# ── _correlation_matrix ───────────────────────────────────────────────────────


class TestCorrelationMatrix:
    def test_single_asset(self):
        ohlcv = {"A": DF_A}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        eq_a = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 100)) * 1000
        corr = bt._correlation_matrix({"A": eq_a})
        assert corr.shape == (1, 1)

    def test_two_assets_shape(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        ea = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 100)) * 1000
        eb = np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 100)) * 1000
        corr = bt._correlation_matrix({"A": ea, "B": eb})
        assert corr.shape == (2, 2)

    def test_diagonal_is_one(self):
        ohlcv = {"A": DF_A, "B": DF_B}
        bt = PortfolioBacktester(ohlcv, _signals(ohlcv))
        ea = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 100)) * 1000
        eb = np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 100)) * 1000
        corr = bt._correlation_matrix({"A": ea, "B": eb})
        assert np.allclose(np.diag(corr), 1.0, atol=1e-9)


# ── run (full) ────────────────────────────────────────────────────────────────


class TestPortfolioRun:
    def _run(self, symbols=None, allocation="equal", allow_short=False):
        symbols = symbols or ["SPY", "QQQ"]
        ohlcv = {"SPY": DF_A, "QQQ": DF_B}
        if len(symbols) == 1:
            ohlcv = {"SPY": DF_A}
        signals = _signals(ohlcv)
        cfg = PortfolioConfig(
            initial_capital=100_000,
            allocation=AllocationMethod(allocation),
            allow_short=allow_short,
        )
        return PortfolioBacktester(ohlcv, signals, config=cfg).run()

    def test_returns_portfolio_result(self):
        r = self._run()
        assert isinstance(r, PortfolioResult)

    def test_symbols_set(self):
        r = self._run()
        assert set(r.symbols) == {"SPY", "QQQ"}

    def test_initial_capital(self):
        r = self._run()
        assert r.initial_capital == 100_000

    def test_final_capital_positive(self):
        r = self._run()
        assert r.final_capital > 0

    def test_equity_curve_length(self):
        r = self._run()
        assert len(r.portfolio_equity) == len(DF_A)

    def test_asset_results_keys(self):
        r = self._run()
        assert "SPY" in r.asset_results
        assert "QQQ" in r.asset_results

    def test_elapsed_ms_positive(self):
        r = self._run()
        assert r.elapsed_ms >= 0.0

    def test_max_drawdown_non_negative(self):
        r = self._run()
        assert r.max_drawdown >= 0.0

    def test_correlation_matrix_shape(self):
        r = self._run()
        assert r.correlation_matrix.shape == (2, 2)

    def test_single_asset(self):
        r = self._run(symbols=["SPY"])
        assert "SPY" in r.symbols

    def test_risk_parity_allocation(self):
        r = self._run(allocation="risk_parity")
        assert r.initial_capital == 100_000

    def test_allow_short(self):
        r = self._run(allow_short=True)
        assert isinstance(r, PortfolioResult)


# ── run_portfolio_backtest ────────────────────────────────────────────────────


class TestRunPortfolioBacktest:
    def test_returns_portfolio_result(self):
        ohlcv = {"SPY": DF_A, "QQQ": DF_B}
        r = run_portfolio_backtest(ohlcv, _signals(ohlcv))
        assert isinstance(r, PortfolioResult)

    def test_initial_capital_propagated(self):
        ohlcv = {"SPY": DF_A}
        r = run_portfolio_backtest(ohlcv, _signals(ohlcv), initial_capital=42_000)
        assert r.initial_capital == 42_000

    def test_equal_allocation(self):
        ohlcv = {"SPY": DF_A, "QQQ": DF_B}
        r = run_portfolio_backtest(ohlcv, _signals(ohlcv), allocation="equal")
        assert isinstance(r, PortfolioResult)

    def test_risk_parity_allocation(self):
        ohlcv = {"SPY": DF_A, "QQQ": DF_B}
        r = run_portfolio_backtest(ohlcv, _signals(ohlcv), allocation="risk_parity")
        assert isinstance(r, PortfolioResult)

    def test_fixed_allocation(self):
        ohlcv = {"SPY": DF_A, "QQQ": DF_B}
        r = run_portfolio_backtest(ohlcv, _signals(ohlcv), allocation="fixed",
                                   fixed_weights={"SPY": 0.6, "QQQ": 0.4})
        assert isinstance(r, PortfolioResult)

    def test_summary_non_empty(self):
        ohlcv = {"SPY": DF_A}
        r = run_portfolio_backtest(ohlcv, _signals(ohlcv))
        assert len(r.summary()) > 0

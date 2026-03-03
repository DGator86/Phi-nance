"""
tests/unit/test_phase6.py
==========================

Unit tests for Phase 6 additions:
  - phinance.data.features       (build_feature_matrix, drop_warmup, feature_names)
  - phinance.backtest.monte_carlo (run_monte_carlo, MCResult)
  - phinance.optimization.walk_forward (WFOResult, WFOWindow, walk_forward_optimize)
  - phinance.strategies.base     (_normalize_abs sign preservation)
  - phinance.__init__            (top-level re-exports)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    """Tests for phinance.data.features.build_feature_matrix."""

    def test_returns_dataframe_with_same_index(self):
        from phinance.data.features import build_feature_matrix
        df = make_ohlcv(100)
        X  = build_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(df)
        assert X.index.equals(df.index)

    def test_all_features_bounded_after_warmup(self):
        """After dropping warmup rows all values should be finite and in a reasonable range."""
        from phinance.data.features import build_feature_matrix, drop_warmup
        df = make_ohlcv(200)
        X  = drop_warmup(build_feature_matrix(df))
        # No all-NaN columns
        for col in X.columns:
            assert not X[col].isna().all(), f"Column {col} is all NaN"
        # All finite values
        assert np.isfinite(X.fillna(0).values).all(), "Non-finite values found in feature matrix"

    def test_rsi_feature_bounded_0_to_1(self):
        from phinance.data.features import build_feature_matrix, drop_warmup
        df = make_ohlcv(100)
        X  = drop_warmup(build_feature_matrix(df))
        rsi_col = X["rsi"].dropna()
        assert (rsi_col >= 0.0).all(), "RSI feature below 0"
        assert (rsi_col <= 1.0).all(), "RSI feature above 1"

    def test_volume_ratio_positive(self):
        from phinance.data.features import build_feature_matrix, drop_warmup
        df = make_ohlcv(100)
        X  = drop_warmup(build_feature_matrix(df))
        vol = X["volume_ratio"].dropna()
        assert (vol >= 0.0).all(), "volume_ratio has negative values"

    def test_calendar_dummies_added_when_requested(self):
        from phinance.data.features import build_feature_matrix
        df = make_ohlcv(100)
        X  = build_feature_matrix(df, include_calendar=True)
        assert "dow_0" in X.columns
        assert "month_1" in X.columns

    def test_feature_names_matches_columns(self):
        from phinance.data.features import build_feature_matrix, feature_names
        df    = make_ohlcv(100)
        X     = build_feature_matrix(df)
        names = feature_names()
        for n in names:
            assert n in X.columns, f"Expected feature '{n}' not in matrix"

    def test_drop_warmup_removes_rows(self):
        from phinance.data.features import build_feature_matrix, drop_warmup, FEATURE_WARMUP_BARS
        df = make_ohlcv(100)
        X  = build_feature_matrix(df)
        Xd = drop_warmup(X)
        assert len(Xd) == len(X) - FEATURE_WARMUP_BARS


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Monte Carlo Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:
    """Tests for phinance.backtest.monte_carlo.run_monte_carlo."""

    def _get_backtest_result(self):
        """Run a quick backtest to get a real BacktestResult."""
        from phinance.backtest.runner import run_backtest
        df = make_ohlcv(200)
        return run_backtest(
            ohlcv=df,
            symbol="TEST",
            indicators={"RSI": {"enabled": True, "params": {"period": 14}}},
            initial_capital=100_000,
        )

    def test_bootstrap_returns_mc_result(self):
        from phinance.backtest.monte_carlo import run_monte_carlo
        br = self._get_backtest_result()
        mc = run_monte_carlo(br, n_simulations=50, method="bootstrap_trades", seed=42)
        assert mc.n_simulations == 50
        assert len(mc.total_return_dist) == 50

    def test_return_shuffle_returns_mc_result(self):
        from phinance.backtest.monte_carlo import run_monte_carlo
        br = self._get_backtest_result()
        mc = run_monte_carlo(br, n_simulations=50, method="return_shuffle", seed=42)
        assert len(mc.sharpe_dist) == 50

    def test_summary_keys_present(self):
        from phinance.backtest.monte_carlo import run_monte_carlo
        br = self._get_backtest_result()
        mc = run_monte_carlo(br, n_simulations=30, seed=0)
        s  = mc.summary
        for key in ["method", "n_simulations", "sharpe", "total_return",
                    "max_drawdown", "prob_positive_return", "prob_positive_sharpe"]:
            assert key in s, f"Key '{key}' missing from MC summary"

    def test_prob_in_0_to_1(self):
        from phinance.backtest.monte_carlo import run_monte_carlo
        br = self._get_backtest_result()
        mc = run_monte_carlo(br, n_simulations=30, seed=1)
        assert 0.0 <= mc.summary["prob_positive_return"] <= 1.0
        assert 0.0 <= mc.summary["prob_positive_sharpe"] <= 1.0

    def test_to_dataframe_shape(self):
        from phinance.backtest.monte_carlo import run_monte_carlo
        br = self._get_backtest_result()
        mc = run_monte_carlo(br, n_simulations=20, seed=2)
        df = mc.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert "sharpe" in df.columns

    def test_quick_sharpe_zero_std(self):
        """_quick_sharpe should return 0 for constant portfolio."""
        from phinance.backtest.monte_carlo import _quick_sharpe
        pv = np.array([100_000.0] * 20)
        assert _quick_sharpe(pv) == 0.0

    def test_quick_max_dd_uptrend(self):
        """Max drawdown of a monotonically rising portfolio is 0."""
        from phinance.backtest.monte_carlo import _quick_max_dd
        pv = np.linspace(100_000, 120_000, 100)
        assert _quick_max_dd(pv) == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Walk-Forward Optimisation
# ─────────────────────────────────────────────────────────────────────────────

class TestWalkForwardOptimize:
    """Tests for phinance.optimization.walk_forward.walk_forward_optimize."""

    def _make_long_ohlcv(self, n: int = 400) -> pd.DataFrame:
        close  = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        close  = np.clip(close, 1.0, None)
        return pd.DataFrame({
            "open":   close * 0.999,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        }, index=pd.date_range("2020-01-01", periods=n))

    def test_wfo_returns_wfo_result(self):
        from phinance.optimization.walk_forward import walk_forward_optimize, WFOResult
        np.random.seed(42)
        df = self._make_long_ohlcv(400)
        result = walk_forward_optimize(
            ohlcv=df,
            symbol="TST",
            in_window=120, out_window=40, step_bars=40,
            n_trials=5,
        )
        assert isinstance(result, WFOResult)

    def test_wfo_creates_multiple_windows(self):
        from phinance.optimization.walk_forward import walk_forward_optimize
        np.random.seed(0)
        df = self._make_long_ohlcv(400)
        result = walk_forward_optimize(
            ohlcv=df,
            symbol="TST",
            in_window=120, out_window=40, step_bars=40,
            n_trials=5,
        )
        assert result.n_windows >= 2

    def test_wfo_summary_keys(self):
        from phinance.optimization.walk_forward import walk_forward_optimize
        np.random.seed(1)
        df = self._make_long_ohlcv(300)
        result = walk_forward_optimize(
            ohlcv=df, symbol="TST",
            in_window=100, out_window=40, step_bars=100,
            n_trials=3,
        )
        s = result.summary
        for key in ["n_windows", "mean_oos_return", "mean_oos_sharpe", "consistency_ratio"]:
            assert key in s, f"Key '{key}' missing from WFO summary"

    def test_wfo_to_dataframe(self):
        from phinance.optimization.walk_forward import walk_forward_optimize
        np.random.seed(2)
        df = self._make_long_ohlcv(300)
        result = walk_forward_optimize(
            ohlcv=df, symbol="TST",
            in_window=100, out_window=40, step_bars=100,
            n_trials=3,
        )
        dff = result.to_dataframe()
        assert isinstance(dff, pd.DataFrame)
        assert len(dff) == result.n_windows


# ─────────────────────────────────────────────────────────────────────────────
#  4.  BaseIndicator._normalize_abs  (sign-preserving normalisation)
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeAbs:
    """Ensure _normalize_abs preserves sign direction."""

    def _normalizer(self):
        from phinance.strategies.base import BaseIndicator

        class _Dummy(BaseIndicator):
            name = "_Dummy"
            def compute(self, df, **_): ...

        return _Dummy()

    def test_positive_input_gives_positive_output(self):
        ind = self._normalizer()
        s   = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        out = ind._normalize_abs(s)
        assert (out > 0).all(), "Positive input should yield positive output"

    def test_negative_input_gives_negative_output(self):
        ind = self._normalizer()
        s   = pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5])
        out = ind._normalize_abs(s)
        assert (out < 0).all(), "Negative input should yield negative output"

    def test_output_bounded_minus1_plus1(self):
        ind = self._normalizer()
        s   = pd.Series(np.linspace(-10, 10, 50))
        out = ind._normalize_abs(s)
        assert (out >= -1.0).all()
        assert (out <= 1.0).all()

    def test_all_nan_returns_zeros(self):
        ind = self._normalizer()
        s   = pd.Series([np.nan, np.nan, np.nan])
        out = ind._normalize_abs(s)
        assert (out == 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Top-level phinance package exports
# ─────────────────────────────────────────────────────────────────────────────

class TestTopLevelExports:
    """Verify the phinance package exposes all declared public symbols."""

    EXPECTED = [
        "fetch_and_cache", "get_cached_dataset", "list_cached_datasets",
        "build_feature_matrix", "drop_warmup", "feature_names",
        "blend_signals", "RunConfig", "RunHistory",
        "run_backtest", "run_monte_carlo", "walk_forward_optimize",
        "list_indicators", "compute_indicator", "INDICATOR_CATALOG",
    ]

    def test_all_symbols_importable(self):
        import phinance
        for sym in self.EXPECTED:
            assert hasattr(phinance, sym), f"phinance.{sym} not found"

    def test_version_string(self):
        import phinance
        assert isinstance(phinance.__version__, str)
        assert phinance.__version__.startswith("1.")

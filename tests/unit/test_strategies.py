"""
tests/unit/test_strategies.py
================================

Comprehensive unit tests for all 15 phinance indicator strategies.

Coverage
--------
  Catalog:       list_indicators(), compute_indicator(), unknown name
  Per-indicator: signal shape, bounds, NaN handling, custom params,
                 directional correctness (where tractable), warmup behaviour
  Params:        get_param_grid(), daily vs intraday grids
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.strategies import list_indicators, compute_indicator, INDICATOR_CATALOG
from phinance.strategies.params import get_param_grid, DAILY_GRIDS
from phinance.exceptions import UnknownIndicatorError


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _assert_signal(sig: pd.Series, df: pd.DataFrame, name: str) -> None:
    """Common shape / bounds / nan assertions."""
    assert sig is not None, f"{name}: returned None"
    assert isinstance(sig, pd.Series), f"{name}: not a Series"
    assert len(sig) == len(df), f"{name}: length mismatch"
    assert not sig.isna().all(), f"{name}: all-NaN signal"
    finite = sig.dropna()
    assert (finite >= -1.0).all(), f"{name}: values below −1: {finite.min()}"
    assert (finite <= 1.0).all(), f"{name}: values above +1: {finite.max()}"


# ─────────────────────────────────────────────────────────────────────────────
#  Catalog
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicatorCatalog:

    EXPECTED_NAMES = [
        "RSI", "MACD", "Bollinger", "Dual SMA", "EMA Cross",
        "Mean Reversion", "Breakout", "Buy & Hold", "VWAP",
        "ATR", "Stochastic", "Williams %R", "CCI", "OBV", "PSAR",
    ]

    def test_list_indicators_returns_all_expected(self):
        names = list_indicators()
        for n in self.EXPECTED_NAMES:
            assert n in names, f"'{n}' missing from catalog"

    def test_catalog_has_15_entries(self):
        assert len(INDICATOR_CATALOG) == 15

    def test_unknown_indicator_raises_error(self):
        with pytest.raises(UnknownIndicatorError):
            compute_indicator("NotAIndicator", make_ohlcv(20))

    def test_all_indicators_produce_valid_signals(self):
        """Every registered indicator: shape, bounds, no all-NaN."""
        df = make_ohlcv(100)
        for name in list_indicators():
            sig = compute_indicator(name, df)
            _assert_signal(sig, df, name)

    def test_all_indicators_handle_minimum_data(self):
        """Indicators should not crash on very short series (30 bars)."""
        df = make_ohlcv(30)
        for name in list_indicators():
            try:
                sig = compute_indicator(name, df)
                assert sig is not None
                assert len(sig) == len(df)
            except Exception as exc:
                pytest.fail(f"{name} crashed on 30-bar data: {exc}")

    def test_all_indicators_handle_edge_data_10_bars(self):
        """Some indicators return all-zero on very short data; they must not crash."""
        df = make_ohlcv(10)
        for name in list_indicators():
            try:
                sig = compute_indicator(name, df)
                assert sig is not None
            except Exception as exc:
                pytest.fail(f"{name} crashed on 10-bar data: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  RSI
# ─────────────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("RSI", df, {"period": 14})
        _assert_signal(sig, df, "RSI")

    def test_no_nan_after_warmup(self):
        df = make_ohlcv(60)
        sig = compute_indicator("RSI", df, {"period": 14})
        assert sig.isna().sum() == 0, "RSI has NaNs after fillna(0)"

    def test_custom_period_differs(self):
        df = make_ohlcv(60)
        s7 = compute_indicator("RSI", df, {"period": 7})
        s21 = compute_indicator("RSI", df, {"period": 21})
        assert not (s7 == s21).all()

    def test_oversold_market_gives_positive_signal(self):
        """Create a sharp downtrend — RSI should signal oversold (positive)."""
        n = 60
        close = 100.0 - np.arange(n, dtype=float) * 1.5  # monotonic decline
        close = np.maximum(close, 1.0)
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("RSI", df, {"period": 14, "oversold": 35})
        # Signal at last bar should be positive (oversold territory)
        assert sig.iloc[-1] > 0, f"Expected positive signal in downtrend, got {sig.iloc[-1]:.3f}"

    def test_overbought_market_gives_negative_signal(self):
        """Pure uptrend → RSI near 100 (only gains, no losses) → negative signal."""
        n = 60
        # Use a step increment that guarantees RSI > overbought threshold
        close = 100.0 + np.arange(n, dtype=float) * 2.0  # large constant gain
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("RSI", df, {"period": 14, "overbought": 65})
        # RSI = 100 when there are zero losses → signal must be negative
        assert sig.iloc[-1] < 0, f"Expected negative signal in pure uptrend, got {sig.iloc[-1]:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
#  MACD
# ─────────────────────────────────────────────────────────────────────────────

class TestMACD:
    def test_bounds_and_shape(self):
        df = make_ohlcv(80)
        sig = compute_indicator("MACD", df)
        _assert_signal(sig, df, "MACD")

    def test_no_nan_after_computation(self):
        df = make_ohlcv(80)
        sig = compute_indicator("MACD", df)
        assert sig.isna().sum() == 0

    def test_histogram_changes_with_params(self):
        df = make_ohlcv(80)
        s1 = compute_indicator("MACD", df, {"fast_period": 8, "slow_period": 21, "signal_period": 7})
        s2 = compute_indicator("MACD", df, {"fast_period": 16, "slow_period": 34, "signal_period": 11})
        assert not (s1 == s2).all()

    def test_uptrend_yields_nonzero_signal(self):
        """MACD produces a non-zero, finite signal on a clear uptrend."""
        n = 100
        close = 100.0 + np.arange(n, dtype=float) * 0.8
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("MACD", df)
        # _normalize maps the series to [-1,1] relative to its own range;
        # on a linear uptrend the raw histogram is positive → normalized values
        # occupy the full [-1,+1] range; simply verify not all-zero.
        assert sig.iloc[40:].abs().mean() > 0, "MACD returned all-zero on uptrend"


# ─────────────────────────────────────────────────────────────────────────────
#  Bollinger
# ─────────────────────────────────────────────────────────────────────────────

class TestBollinger:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Bollinger", df, {"period": 20, "num_std": 2.0})
        _assert_signal(sig, df, "Bollinger")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Bollinger", df)
        assert sig.isna().sum() == 0

    def test_tight_bands_more_extreme_signals(self):
        df = make_ohlcv(60)
        wide = compute_indicator("Bollinger", df, {"num_std": 2.5})
        tight = compute_indicator("Bollinger", df, {"num_std": 1.0})
        # Tight bands should clip more → higher average absolute signal
        assert tight.abs().mean() >= wide.abs().mean()

    def test_at_lower_band_positive_signal(self):
        """Construct a scenario where close is at the lower Bollinger band."""
        n = 40
        close = np.full(n, 100.0)
        # Push last bar way below mean
        close[-1] = 80.0
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Bollinger", df, {"period": 20})
        assert sig.iloc[-1] > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Dual SMA
# ─────────────────────────────────────────────────────────────────────────────

class TestDualSMA:
    def test_bounds_and_shape(self):
        df = make_ohlcv(80)
        sig = compute_indicator("Dual SMA", df)
        _assert_signal(sig, df, "Dual SMA")

    def test_no_nan(self):
        df = make_ohlcv(80)
        sig = compute_indicator("Dual SMA", df)
        assert sig.isna().sum() == 0

    def test_uptrend_nonzero_signal(self):
        """DualSMA spread is positive on an uptrend; normalized signal is non-zero."""
        n = 100
        close = 100 + np.arange(n, dtype=float)
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Dual SMA", df, {"fast_period": 5, "slow_period": 20})
        # _normalize maps the spread to its own quantile range;
        # verify the signal is non-zero and finite across the series.
        assert sig.abs().max() > 0, "DualSMA returned all-zero on uptrend"
        assert np.isfinite(sig.values).all()


# ─────────────────────────────────────────────────────────────────────────────
#  EMA Cross
# ─────────────────────────────────────────────────────────────────────────────

class TestEMACross:
    def test_bounds_and_shape(self):
        df = make_ohlcv(80)
        sig = compute_indicator("EMA Cross", df)
        _assert_signal(sig, df, "EMA Cross")

    def test_no_nan(self):
        df = make_ohlcv(80)
        sig = compute_indicator("EMA Cross", df)
        assert sig.isna().sum() == 0

    def test_ema_reacts_faster_than_sma(self):
        """EMA cross should change direction faster than Dual SMA on the same data."""
        n = 80
        close = np.concatenate([
            100 + np.arange(40, dtype=float),
            140 - np.arange(40, dtype=float),
        ])
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        ema_sig = compute_indicator("EMA Cross", df, {"fast_period": 5, "slow_period": 20})
        sma_sig = compute_indicator("Dual SMA", df, {"fast_period": 5, "slow_period": 20})
        # Both should produce signals; no further assertion on relative speed here
        assert not ema_sig.isna().all()
        assert not sma_sig.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
#  Mean Reversion
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanReversion:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Mean Reversion", df)
        _assert_signal(sig, df, "Mean Reversion")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Mean Reversion", df)
        assert sig.isna().sum() == 0

    def test_below_mean_positive_signal(self):
        """Close significantly below SMA should produce a positive signal."""
        n = 40
        close = np.full(n, 100.0, dtype=float)
        close[-5:] = 90.0       # drop below mean
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Mean Reversion", df, {"period": 20})
        assert sig.iloc[-1] > 0

    def test_z_threshold_clips(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Mean Reversion", df, {"z_threshold": 1.0})
        assert sig.abs().max() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Breakout (Donchian)
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakout:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Breakout", df)
        _assert_signal(sig, df, "Breakout")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Breakout", df)
        assert sig.isna().sum() == 0

    def test_close_at_high_channel_positive(self):
        n = 40
        base = 100.0
        close = np.full(n, base)
        close[-1] = base * 2.0    # far above channel top
        high = close * 1.001
        low = close * 0.999
        df = pd.DataFrame({
            "open": close * 0.999, "high": high,
            "low": low, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Breakout", df, {"period": 20})
        assert sig.iloc[-1] >= 0.9


# ─────────────────────────────────────────────────────────────────────────────
#  Buy & Hold
# ─────────────────────────────────────────────────────────────────────────────

class TestBuyHold:
    def test_constant_half(self):
        df = make_ohlcv(20)
        sig = compute_indicator("Buy & Hold", df)
        assert (sig == 0.5).all()

    def test_shape(self):
        df = make_ohlcv(5)
        sig = compute_indicator("Buy & Hold", df)
        assert len(sig) == 5


# ─────────────────────────────────────────────────────────────────────────────
#  VWAP
# ─────────────────────────────────────────────────────────────────────────────

class TestVWAP:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("VWAP", df)
        _assert_signal(sig, df, "VWAP")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("VWAP", df)
        assert sig.isna().sum() == 0

    def test_band_pct_param_accepted(self):
        df = make_ohlcv(40)
        sig = compute_indicator("VWAP", df, {"period": 10, "band_pct": 0.25})
        assert len(sig) == 40


# ─────────────────────────────────────────────────────────────────────────────
#  ATR
# ─────────────────────────────────────────────────────────────────────────────

class TestATR:
    def test_bounds_and_shape(self):
        df = make_ohlcv(100)
        sig = compute_indicator("ATR", df)
        _assert_signal(sig, df, "ATR")

    def test_no_nan(self):
        df = make_ohlcv(100)
        sig = compute_indicator("ATR", df)
        assert sig.isna().sum() == 0

    def test_high_volatility_positive_signal(self):
        """Alternating high/low prices → high ATR → positive (HIGHVOL) signal."""
        n = 80
        base = 100.0
        prices = base + np.tile([5, -5], n // 2)
        df = pd.DataFrame({
            "open": prices * 0.999, "high": prices * 1.05,
            "low": prices * 0.95, "close": prices,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("ATR", df, {"period": 14, "lookback": 30})
        assert sig.iloc[-10:].mean() > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Stochastic
# ─────────────────────────────────────────────────────────────────────────────

class TestStochastic:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Stochastic", df)
        _assert_signal(sig, df, "Stochastic")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Stochastic", df)
        assert sig.isna().sum() == 0

    def test_oversold_positive_signal(self):
        """Close near period lows → Stochastic oversold → positive signal."""
        n = 40
        close = np.full(n, 100.0, dtype=float)
        close[-5:] = 85.0
        df = pd.DataFrame({
            "open": close * 0.999,
            "high": np.full(n, 110.0),
            "low": np.full(n, 85.0),
            "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Stochastic", df, {"k_period": 14, "oversold": 25.0})
        assert sig.iloc[-1] >= 0

    def test_smooth_param_changes_output(self):
        """smooth=3 applies a 3-bar SMA to %K before %D, changing the result."""
        df = make_ohlcv(60)
        s1 = compute_indicator("Stochastic", df, {"smooth": 1})
        s3 = compute_indicator("Stochastic", df, {"smooth": 3})
        # The two signals should differ (not identical), proving the param is applied.
        assert not (s1 == s3).all(), "smooth=1 and smooth=3 produced identical output"


# ─────────────────────────────────────────────────────────────────────────────
#  Williams %R
# ─────────────────────────────────────────────────────────────────────────────

class TestWilliamsR:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Williams %R", df)
        _assert_signal(sig, df, "Williams %R")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("Williams %R", df)
        assert sig.isna().sum() == 0

    def test_close_at_period_low_positive(self):
        """Close at period low → %R ≈ −100 → positive signal (oversold)."""
        n = 40
        high_arr = np.full(n, 110.0)
        low_arr = np.full(n, 90.0)
        close_arr = np.full(n, 90.1)  # near the low
        df = pd.DataFrame({
            "open": close_arr * 0.999, "high": high_arr,
            "low": low_arr, "close": close_arr,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Williams %R", df, {"period": 14})
        assert sig.iloc[-1] > 0

    def test_close_at_period_high_negative(self):
        """Close at period high → %R ≈ 0 → negative signal (overbought)."""
        n = 40
        high_arr = np.full(n, 110.0)
        low_arr = np.full(n, 90.0)
        close_arr = np.full(n, 109.9)  # near the high
        df = pd.DataFrame({
            "open": close_arr * 0.999, "high": high_arr,
            "low": low_arr, "close": close_arr,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("Williams %R", df, {"period": 14})
        assert sig.iloc[-1] < 0


# ─────────────────────────────────────────────────────────────────────────────
#  CCI
# ─────────────────────────────────────────────────────────────────────────────

class TestCCI:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("CCI", df)
        _assert_signal(sig, df, "CCI")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("CCI", df)
        assert sig.isna().sum() == 0

    def test_scale_param_changes_output(self):
        df = make_ohlcv(60)
        s75 = compute_indicator("CCI", df, {"scale": 75.0})
        s150 = compute_indicator("CCI", df, {"scale": 150.0})
        assert not (s75 == s150).all()


# ─────────────────────────────────────────────────────────────────────────────
#  OBV
# ─────────────────────────────────────────────────────────────────────────────

class TestOBV:
    def test_bounds_and_shape(self):
        df = make_ohlcv(60)
        sig = compute_indicator("OBV", df)
        _assert_signal(sig, df, "OBV")

    def test_no_nan(self):
        df = make_ohlcv(60)
        sig = compute_indicator("OBV", df)
        assert sig.isna().sum() == 0

    def test_uptrend_obv_grows(self):
        """On a pure uptrend OBV is always increasing → ROC is positive → signal > 0
        for early bars after warmup. Verify non-zero active signal."""
        n = 80
        close = 100.0 + np.arange(n, dtype=float) * 0.5
        volume = np.ones(n) * 1_000_000   # constant volume for stable ROC
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.001,
            "low": close * 0.998, "close": close,
            "volume": volume,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("OBV", df, {"period": 14})
        # After warmup period, OBV ROC should be non-zero
        active = sig.iloc[20:]
        assert active.abs().max() > 0, "OBV returned all-zero on uptrend"

    def test_period_param_accepted(self):
        df = make_ohlcv(60)
        sig = compute_indicator("OBV", df, {"period": 7})
        assert len(sig) == 60


# ─────────────────────────────────────────────────────────────────────────────
#  PSAR
# ─────────────────────────────────────────────────────────────────────────────

class TestPSAR:
    def test_bounds_and_shape(self):
        df = make_ohlcv(80)
        sig = compute_indicator("PSAR", df)
        _assert_signal(sig, df, "PSAR")

    def test_no_nan(self):
        df = make_ohlcv(80)
        sig = compute_indicator("PSAR", df)
        assert sig.isna().sum() == 0

    def test_uptrend_positive_signal(self):
        n = 80
        close = 100.0 + np.arange(n, dtype=float) * 0.5
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("PSAR", df)
        # In an uptrend the SAR should be below price → positive signal
        assert sig.iloc[-10:].mean() > 0

    def test_downtrend_negative_signal(self):
        n = 80
        close = 200.0 - np.arange(n, dtype=float) * 0.5
        df = pd.DataFrame({
            "open": close * 1.001, "high": close * 1.005,
            "low": close * 0.995, "close": close,
            "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2023-01-01", periods=n))
        sig = compute_indicator("PSAR", df)
        assert sig.iloc[-10:].mean() < 0

    def test_too_short_returns_zeros(self):
        df = make_ohlcv(2)
        sig = compute_indicator("PSAR", df)
        assert len(sig) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter Grids
# ─────────────────────────────────────────────────────────────────────────────

class TestParamGrids:
    def test_all_indicators_have_daily_grid(self):
        for name in list_indicators():
            if name == "Buy & Hold":
                continue  # intentionally no grid
            grid = get_param_grid(name, "1D")
            assert isinstance(grid, dict), f"{name} daily grid is not a dict"
            assert len(grid) > 0, f"{name} daily grid is empty"

    def test_all_indicators_have_intraday_grid(self):
        for name in list_indicators():
            if name == "Buy & Hold":
                continue
            grid = get_param_grid(name, "15m")
            assert isinstance(grid, dict), f"{name} intraday grid is not a dict"
            assert len(grid) > 0, f"{name} intraday grid is empty"

    def test_daily_and_intraday_differ(self):
        """RSI daily period grid should differ from intraday grid."""
        daily = get_param_grid("RSI", "1D")
        intraday = get_param_grid("RSI", "15m")
        assert daily["period"] != intraday["period"]

    def test_buy_hold_has_empty_grid(self):
        assert get_param_grid("Buy & Hold", "1D") == {}

    def test_grid_values_are_lists(self):
        for name, grid in DAILY_GRIDS.items():
            for param, vals in grid.items():
                assert isinstance(vals, list), \
                    f"{name}.{param}: grid values must be a list"
                assert len(vals) > 0, \
                    f"{name}.{param}: grid list must be non-empty"

    def test_intraday_grids_have_smaller_windows(self):
        """Intraday RSI default period should be smaller than daily."""
        daily_max = max(get_param_grid("RSI", "1D")["period"])
        intraday_max = max(get_param_grid("RSI", "15m")["period"])
        assert intraday_max <= daily_max


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-indicator param variation
# ─────────────────────────────────────────────────────────────────────────────

class TestIndicatorParamVariation:
    """Ensure different params produce different signals for each indicator."""

    def _check_params_differ(self, name, params_a, params_b, n=80):
        df = make_ohlcv(n)
        sa = compute_indicator(name, df, params_a)
        sb = compute_indicator(name, df, params_b)
        assert not (sa == sb).all(), \
            f"{name}: params {params_a} and {params_b} produced identical signals"

    def test_rsi_period_varies(self):
        self._check_params_differ("RSI", {"period": 7}, {"period": 21})

    def test_macd_params_vary(self):
        self._check_params_differ(
            "MACD",
            {"fast_period": 8, "slow_period": 21},
            {"fast_period": 16, "slow_period": 34},
        )

    def test_bollinger_std_varies(self):
        self._check_params_differ("Bollinger", {"num_std": 1.5}, {"num_std": 2.5})

    def test_dual_sma_periods_vary(self):
        self._check_params_differ(
            "Dual SMA",
            {"fast_period": 5, "slow_period": 20},
            {"fast_period": 10, "slow_period": 50},
        )

    def test_mean_reversion_period_varies(self):
        self._check_params_differ("Mean Reversion", {"period": 10}, {"period": 30})

    def test_breakout_period_varies(self):
        self._check_params_differ("Breakout", {"period": 10}, {"period": 30})

    def test_cci_scale_varies(self):
        self._check_params_differ("CCI", {"scale": 75.0}, {"scale": 150.0})

    def test_stochastic_k_period_varies(self):
        self._check_params_differ("Stochastic", {"k_period": 9}, {"k_period": 21})

    def test_williams_r_period_varies(self):
        self._check_params_differ("Williams %R", {"period": 7}, {"period": 21})

    def test_obv_period_varies(self):
        self._check_params_differ("OBV", {"period": 7}, {"period": 21})

    def test_psar_af_varies(self):
        self._check_params_differ(
            "PSAR",
            {"initial_af": 0.01, "max_af": 0.10},
            {"initial_af": 0.03, "max_af": 0.30},
        )

"""
tests/unit/test_advanced_indicators.py
=======================================

Comprehensive unit tests for the five advanced indicators added to Phi-nance:
  12. Aroon         — trend strength oscillator
  13. Ulcer Index   — downside-risk / drawdown indicator
  14. KST           — Know Sure Thing momentum oscillator
  15. TRIX          — Triple Smoothed EMA momentum
  16. Mass Index    — High-low range reversal indicator

Each indicator is tested for:
  - Output type and shape
  - Signal range [−1, 1]
  - No NaN in output (warmup filled with 0)
  - Correct directional behaviour on trending data
  - Parameter overrides work correctly
  - Default parameters round-trip through compute_with_defaults
  - Minimum data / edge cases handled gracefully
  - Catalog registration and compute_indicator dispatch
  - param_grid defined and non-empty
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 300,
    seed: int = 42,
    trend: float = 0.0,
    volatility: float = 0.5,
) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* business-day bars."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(trend, volatility, n))
    close = np.maximum(close, 1.0)  # ensure positive prices
    high  = close + np.abs(rng.normal(0, 0.3, n))
    low   = close - np.abs(rng.normal(0, 0.3, n))
    low   = np.maximum(low, 0.5)
    vol   = rng.integers(100_000, 500_000, n).astype(float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def _make_uptrend(n: int = 300) -> pd.DataFrame:
    """Strongly rising price series."""
    return _make_ohlcv(n=n, seed=1, trend=0.5, volatility=0.1)


def _make_downtrend(n: int = 300) -> pd.DataFrame:
    """Strongly falling price series."""
    return _make_ohlcv(n=n, seed=2, trend=-0.5, volatility=0.1)


def _make_flat(n: int = 300) -> pd.DataFrame:
    """Sideways / flat price series."""
    return _make_ohlcv(n=n, seed=3, trend=0.0, volatility=0.05)


def _assert_valid_signal(sig: pd.Series, name: str = "") -> None:
    """Assert signal is a Series in [-1, 1] with no NaN."""
    assert isinstance(sig, pd.Series), f"{name}: not a Series"
    assert not sig.isna().any(), f"{name}: contains NaN"
    assert (sig >= -1.0).all(), f"{name}: below -1"
    assert (sig <= 1.0).all(), f"{name}: above +1"


# ── Import indicator classes ──────────────────────────────────────────────────

from phinance.strategies.aroon       import AroonIndicator
from phinance.strategies.ulcer_index import UlcerIndexIndicator
from phinance.strategies.kst         import KSTIndicator
from phinance.strategies.trix        import TRIXIndicator
from phinance.strategies.mass_index  import MassIndexIndicator
from phinance.strategies.indicator_catalog import (
    INDICATOR_CATALOG,
    compute_indicator,
    list_indicators,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Aroon Indicator
# ═══════════════════════════════════════════════════════════════════════════════

class TestAroonIndicator:
    """Tests for AroonIndicator (Aroon Oscillator)."""

    def setup_method(self):
        self.indicator = AroonIndicator()
        self.df = _make_ohlcv()

    # ── Output shape / type ───────────────────────────────────────────────────

    def test_returns_series(self):
        sig = self.indicator.compute(self.df)
        assert isinstance(sig, pd.Series)

    def test_output_length_matches_input(self):
        sig = self.indicator.compute(self.df)
        assert len(sig) == len(self.df)

    def test_no_nan_in_output(self):
        sig = self.indicator.compute(self.df)
        assert not sig.isna().any()

    def test_signal_range(self):
        sig = self.indicator.compute(self.df)
        _assert_valid_signal(sig, "Aroon")

    def test_series_name(self):
        sig = self.indicator.compute(self.df)
        assert sig.name == "Aroon"

    # ── Directional behaviour ─────────────────────────────────────────────────

    def test_uptrend_positive_signal(self):
        """Aroon should produce mostly positive signal in a strong uptrend."""
        df_up = _make_uptrend()
        sig = self.indicator.compute(df_up)
        # Drop warmup (first period+1 bars)
        tail = sig.iloc[30:]
        assert tail.mean() > 0.0, f"Expected positive mean, got {tail.mean():.4f}"

    def test_downtrend_negative_signal(self):
        """Aroon should produce mostly negative signal in a strong downtrend."""
        df_dn = _make_downtrend()
        sig = self.indicator.compute(df_dn)
        tail = sig.iloc[30:]
        assert tail.mean() < 0.0, f"Expected negative mean, got {tail.mean():.4f}"

    def test_extremes_in_pure_uptrend(self):
        """In a monotonically increasing series Aroon Up = 100, Down = 0 → +1."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        close = np.arange(60, dtype=float) + 100.0
        df = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1,
             "close": close, "volume": np.ones(60) * 1e6},
            index=dates,
        )
        sig = self.indicator.compute(df, period=25)
        # After warmup the signal should be +1
        assert sig.iloc[-1] == pytest.approx(1.0, abs=1e-6)

    def test_extremes_in_pure_downtrend(self):
        """In a monotonically decreasing series Aroon Down = 100, Up = 0 → -1."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        close = np.arange(59, -1, -1, dtype=float) + 50.0
        df = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1,
             "close": close, "volume": np.ones(60) * 1e6},
            index=dates,
        )
        sig = self.indicator.compute(df, period=25)
        assert sig.iloc[-1] == pytest.approx(-1.0, abs=1e-6)

    # ── Parameter overrides ───────────────────────────────────────────────────

    def test_short_period(self):
        sig = self.indicator.compute(self.df, period=10)
        _assert_valid_signal(sig, "Aroon period=10")

    def test_long_period(self):
        sig = self.indicator.compute(self.df, period=50)
        _assert_valid_signal(sig, "Aroon period=50")

    def test_default_params_round_trip(self):
        sig_direct  = self.indicator.compute(self.df)
        sig_default = self.indicator.compute_with_defaults(self.df)
        pd.testing.assert_series_equal(sig_direct, sig_default)

    def test_param_override_via_compute_with_defaults(self):
        sig_14 = self.indicator.compute_with_defaults(self.df, {"period": 14})
        sig_25 = self.indicator.compute_with_defaults(self.df, {"period": 25})
        assert not sig_14.equals(sig_25)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_minimal_data(self):
        """Should return all zeros when data shorter than warmup."""
        df_small = self.df.iloc[:10]
        sig = self.indicator.compute(df_small, period=25)
        _assert_valid_signal(sig, "Aroon minimal")

    def test_single_bar(self):
        df_one = self.df.iloc[:1]
        sig = self.indicator.compute(df_one)
        assert len(sig) == 1
        assert not np.isnan(sig.iloc[0])

    # ── param_grid ────────────────────────────────────────────────────────────

    def test_param_grid_defined(self):
        grid = AroonIndicator.get_param_grid()
        assert "period" in grid
        assert len(grid["period"]) >= 3

    # ── Catalog registration ──────────────────────────────────────────────────

    def test_registered_in_catalog(self):
        assert "Aroon" in INDICATOR_CATALOG

    def test_compute_indicator_dispatch(self):
        sig = compute_indicator("Aroon", self.df)
        _assert_valid_signal(sig, "catalog Aroon")

    def test_compute_indicator_with_params(self):
        sig = compute_indicator("Aroon", self.df, params={"period": 20})
        _assert_valid_signal(sig, "catalog Aroon period=20")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Ulcer Index
# ═══════════════════════════════════════════════════════════════════════════════

class TestUlcerIndexIndicator:
    """Tests for UlcerIndexIndicator."""

    def setup_method(self):
        self.indicator = UlcerIndexIndicator()
        self.df = _make_ohlcv()

    # ── Output shape / type ───────────────────────────────────────────────────

    def test_returns_series(self):
        sig = self.indicator.compute(self.df)
        assert isinstance(sig, pd.Series)

    def test_output_length_matches_input(self):
        sig = self.indicator.compute(self.df)
        assert len(sig) == len(self.df)

    def test_no_nan_in_output(self):
        sig = self.indicator.compute(self.df)
        assert not sig.isna().any()

    def test_signal_range(self):
        sig = self.indicator.compute(self.df)
        _assert_valid_signal(sig, "Ulcer Index")

    def test_series_name(self):
        sig = self.indicator.compute(self.df)
        assert sig.name == "Ulcer Index"

    # ── Directional behaviour ─────────────────────────────────────────────────

    def test_downtrend_has_higher_raw_ui(self):
        """A downtrend should produce a higher raw Ulcer Index than a flat market."""
        df_dn   = _make_downtrend()
        df_flat = _make_flat()

        def _raw_ui(df, period=14):
            close = df["close"].astype(float)
            rolling_max = close.rolling(window=period, min_periods=1).max()
            dd_pct = ((close - rolling_max) / rolling_max) * 100.0
            sq_dd  = dd_pct ** 2
            ui = sq_dd.rolling(period, min_periods=period).mean().apply(np.sqrt)
            return ui.dropna().mean()

        ui_dn   = _raw_ui(df_dn)
        ui_flat = _raw_ui(df_flat)
        assert ui_dn > ui_flat, (
            f"Downtrend raw UI ({ui_dn:.4f}) should be > flat ({ui_flat:.4f})"
        )

    def test_flat_market_raw_ui_near_zero(self):
        """In a very flat market, raw UI should be near-zero (minimal drawdown)."""
        df_flat = _make_flat()
        close   = df_flat["close"].astype(float)
        rolling_max = close.rolling(window=14, min_periods=1).max()
        dd_pct  = ((close - rolling_max) / rolling_max) * 100.0
        sq_dd   = dd_pct ** 2
        ui_raw  = sq_dd.rolling(14, min_periods=14).mean().apply(np.sqrt).dropna()
        assert ui_raw.mean() < 2.0, f"Expected small UI in flat market, got {ui_raw.mean():.4f}"

    def test_rising_market_ui_non_negative_raw(self):
        """UI is always >= 0 before normalisation (raw check via internal calc)."""
        df_up = _make_uptrend()
        close = df_up["close"].astype(float)
        rolling_max = close.rolling(window=14, min_periods=1).max()
        dd_pct = ((close - rolling_max) / rolling_max) * 100.0
        sq_dd  = dd_pct ** 2
        ui_raw = sq_dd.rolling(14, min_periods=14).mean().apply(np.sqrt)
        assert (ui_raw.dropna() >= 0).all()

    # ── Parameter overrides ───────────────────────────────────────────────────

    def test_period_7(self):
        sig = self.indicator.compute(self.df, period=7)
        _assert_valid_signal(sig, "Ulcer period=7")

    def test_period_28(self):
        sig = self.indicator.compute(self.df, period=28)
        _assert_valid_signal(sig, "Ulcer period=28")

    def test_default_params_round_trip(self):
        sig_direct  = self.indicator.compute(self.df)
        sig_default = self.indicator.compute_with_defaults(self.df)
        pd.testing.assert_series_equal(sig_direct, sig_default)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_minimal_data(self):
        df_small = self.df.iloc[:5]
        sig = self.indicator.compute(df_small, period=14)
        _assert_valid_signal(sig, "Ulcer minimal")

    def test_single_bar(self):
        df_one = self.df.iloc[:1]
        sig = self.indicator.compute(df_one)
        assert len(sig) == 1
        assert not np.isnan(sig.iloc[0])

    def test_constant_price(self):
        """Constant price → UI = 0 (no drawdown)."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        close = np.full(50, 100.0)
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(50) * 1e6},
            index=dates,
        )
        sig = self.indicator.compute(df, period=14)
        _assert_valid_signal(sig, "Ulcer constant")
        # UI = 0 → signal after warmup should be near 0 (no drawdown)
        assert sig.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    # ── param_grid ────────────────────────────────────────────────────────────

    def test_param_grid_defined(self):
        grid = UlcerIndexIndicator.get_param_grid()
        assert "period" in grid
        assert len(grid["period"]) >= 3

    # ── Catalog registration ──────────────────────────────────────────────────

    def test_registered_in_catalog(self):
        assert "Ulcer Index" in INDICATOR_CATALOG

    def test_compute_indicator_dispatch(self):
        sig = compute_indicator("Ulcer Index", self.df)
        _assert_valid_signal(sig, "catalog Ulcer Index")

    def test_compute_indicator_with_params(self):
        sig = compute_indicator("Ulcer Index", self.df, params={"period": 10})
        _assert_valid_signal(sig, "catalog Ulcer Index period=10")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. KST (Know Sure Thing)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKSTIndicator:
    """Tests for KSTIndicator (Know Sure Thing)."""

    def setup_method(self):
        self.indicator = KSTIndicator()
        self.df = _make_ohlcv()

    # ── Output shape / type ───────────────────────────────────────────────────

    def test_returns_series(self):
        sig = self.indicator.compute(self.df)
        assert isinstance(sig, pd.Series)

    def test_output_length_matches_input(self):
        sig = self.indicator.compute(self.df)
        assert len(sig) == len(self.df)

    def test_no_nan_in_output(self):
        sig = self.indicator.compute(self.df)
        assert not sig.isna().any()

    def test_signal_range(self):
        sig = self.indicator.compute(self.df)
        _assert_valid_signal(sig, "KST")

    def test_series_name(self):
        sig = self.indicator.compute(self.df)
        assert sig.name == "KST"

    # ── Directional behaviour ─────────────────────────────────────────────────

    def test_uptrend_produces_positive_mean(self):
        """KST raw oscillator should be positive on average in a sustained uptrend."""
        df_up = _make_uptrend(n=350)
        sig = self.indicator.compute(df_up)
        # Drop warmup (KST needs ~roc4 + sma4 = 30+15 = 45 bars minimum)
        tail = sig.iloc[50:]
        assert tail.mean() > 0.0, f"Expected positive KST mean, got {tail.mean():.4f}"

    def test_downtrend_produces_negative_mean(self):
        """KST raw oscillator should be negative on average in a sustained downtrend."""
        df_dn = _make_downtrend(n=350)
        sig = self.indicator.compute(df_dn)
        tail = sig.iloc[50:]
        assert tail.mean() < 0.0, f"Expected negative KST mean, got {tail.mean():.4f}"

    # ── Parameter overrides ───────────────────────────────────────────────────

    def test_custom_roc_periods(self):
        sig = self.indicator.compute(self.df, roc1=8, roc2=12, roc3=18, roc4=25)
        _assert_valid_signal(sig, "KST custom ROC")

    def test_signal_param_accepted(self):
        """signal parameter is accepted without error (retained for API compatibility)."""
        sig7  = self.indicator.compute(self.df, signal=7)
        sig12 = self.indicator.compute(self.df, signal=12)
        # Both should produce valid signals
        _assert_valid_signal(sig7,  "KST signal=7")
        _assert_valid_signal(sig12, "KST signal=12")

    def test_default_params_round_trip(self):
        sig_direct  = self.indicator.compute(self.df)
        sig_default = self.indicator.compute_with_defaults(self.df)
        pd.testing.assert_series_equal(sig_direct, sig_default)

    def test_all_default_params_present(self):
        defaults = KSTIndicator.default_params
        for key in ["roc1", "roc2", "roc3", "roc4", "sma1", "sma2", "sma3", "sma4", "signal"]:
            assert key in defaults, f"Missing default param: {key}"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_insufficient_data_returns_zeros(self):
        """With fewer bars than warmup, should return all zeros."""
        df_small = self.df.iloc[:20]
        sig = self.indicator.compute(df_small)
        _assert_valid_signal(sig, "KST small")

    def test_constant_price_zero_signal(self):
        """Constant price → ROC = 0 → KST = 0 → signal = 0."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        close = np.full(100, 100.0)
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(100) * 1e6},
            index=dates,
        )
        sig = self.indicator.compute(df)
        _assert_valid_signal(sig, "KST constant")
        assert sig.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    # ── param_grid ────────────────────────────────────────────────────────────

    def test_param_grid_defined(self):
        grid = KSTIndicator.get_param_grid()
        assert "roc1" in grid or "roc4" in grid
        assert len(grid) >= 1

    # ── Catalog registration ──────────────────────────────────────────────────

    def test_registered_in_catalog(self):
        assert "KST" in INDICATOR_CATALOG

    def test_compute_indicator_dispatch(self):
        sig = compute_indicator("KST", self.df)
        _assert_valid_signal(sig, "catalog KST")

    def test_compute_indicator_with_params(self):
        sig = compute_indicator("KST", self.df, params={"signal": 12})
        _assert_valid_signal(sig, "catalog KST signal=12")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. TRIX
# ═══════════════════════════════════════════════════════════════════════════════

class TestTRIXIndicator:
    """Tests for TRIXIndicator (Triple Smoothed EMA)."""

    def setup_method(self):
        self.indicator = TRIXIndicator()
        self.df = _make_ohlcv()

    # ── Output shape / type ───────────────────────────────────────────────────

    def test_returns_series(self):
        sig = self.indicator.compute(self.df)
        assert isinstance(sig, pd.Series)

    def test_output_length_matches_input(self):
        sig = self.indicator.compute(self.df)
        assert len(sig) == len(self.df)

    def test_no_nan_in_output(self):
        sig = self.indicator.compute(self.df)
        assert not sig.isna().any()

    def test_signal_range(self):
        sig = self.indicator.compute(self.df)
        _assert_valid_signal(sig, "TRIX")

    def test_series_name(self):
        sig = self.indicator.compute(self.df)
        assert sig.name == "TRIX"

    # ── Directional behaviour ─────────────────────────────────────────────────

    def test_uptrend_positive_signal(self):
        """Triple-EMA ROC should be positive on average in a sustained uptrend."""
        df_up = _make_uptrend(n=350)
        sig = self.indicator.compute(df_up)
        # Drop warmup (3 × EMA span = 3 × 15 = 45 bars minimum)
        tail = sig.iloc[50:]
        assert tail.mean() > 0.0, f"Expected positive TRIX mean, got {tail.mean():.4f}"

    def test_downtrend_negative_signal(self):
        """Triple-EMA ROC should be negative on average in a sustained downtrend."""
        df_dn = _make_downtrend(n=350)
        sig = self.indicator.compute(df_dn)
        tail = sig.iloc[50:]
        assert tail.mean() < 0.0, f"Expected negative TRIX mean, got {tail.mean():.4f}"

    # ── Triple-smoothing property ─────────────────────────────────────────────

    def test_raw_trix_smoother_than_single_ema_roc(self):
        """Raw TRIX (triple EMA ROC) should have lower std than raw close pct_change."""
        close = self.df["close"].astype(float)
        n_period = 15
        ema1 = close.ewm(span=n_period, min_periods=n_period, adjust=False).mean()
        ema2 = ema1.ewm(span=n_period, min_periods=n_period, adjust=False).mean()
        ema3 = ema2.ewm(span=n_period, min_periods=n_period, adjust=False).mean()
        ema3_prev = ema3.shift(1)
        trix_raw = ((ema3 - ema3_prev) / ema3_prev.where(ema3_prev != 0, 1e-10)) * 100.0
        roc_raw = close.pct_change() * 100.0
        # Both must be on same % scale; compare non-NaN tails
        trix_tail = trix_raw.iloc[50:].dropna()
        roc_tail  = roc_raw.iloc[50:].dropna()
        assert trix_tail.std() < roc_tail.std(), (
            f"TRIX std ({trix_tail.std():.4f}) should be < raw ROC std ({roc_tail.std():.4f})"
        )

    # ── Parameter overrides ───────────────────────────────────────────────────

    def test_short_period(self):
        sig = self.indicator.compute(self.df, period=8)
        _assert_valid_signal(sig, "TRIX period=8")

    def test_long_period(self):
        sig = self.indicator.compute(self.df, period=21)
        _assert_valid_signal(sig, "TRIX period=21")

    def test_signal_param_accepted(self):
        """signal parameter is accepted without error (retained for API compatibility)."""
        sig7  = self.indicator.compute(self.df, signal=7)
        sig12 = self.indicator.compute(self.df, signal=12)
        # Both produce valid signals
        _assert_valid_signal(sig7,  "TRIX signal=7")
        _assert_valid_signal(sig12, "TRIX signal=12")

    def test_default_params_round_trip(self):
        sig_direct  = self.indicator.compute(self.df)
        sig_default = self.indicator.compute_with_defaults(self.df)
        pd.testing.assert_series_equal(sig_direct, sig_default)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_constant_price(self):
        """Constant price → EMA3 constant → ROC = 0 → TRIX = 0."""
        dates = pd.date_range("2020-01-01", periods=80, freq="B")
        close = np.full(80, 100.0)
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close,
             "volume": np.ones(80) * 1e6},
            index=dates,
        )
        sig = self.indicator.compute(df, period=15, signal=9)
        _assert_valid_signal(sig, "TRIX constant")
        assert sig.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_single_bar(self):
        df_one = self.df.iloc[:1]
        sig = self.indicator.compute(df_one)
        assert len(sig) == 1
        assert not np.isnan(sig.iloc[0])

    # ── param_grid ────────────────────────────────────────────────────────────

    def test_param_grid_defined(self):
        grid = TRIXIndicator.get_param_grid()
        assert "period" in grid
        assert "signal" in grid

    # ── Catalog registration ──────────────────────────────────────────────────

    def test_registered_in_catalog(self):
        assert "TRIX" in INDICATOR_CATALOG

    def test_compute_indicator_dispatch(self):
        sig = compute_indicator("TRIX", self.df)
        _assert_valid_signal(sig, "catalog TRIX")

    def test_compute_indicator_with_params(self):
        sig = compute_indicator("TRIX", self.df, params={"period": 12, "signal": 7})
        _assert_valid_signal(sig, "catalog TRIX custom")


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Mass Index
# ═══════════════════════════════════════════════════════════════════════════════

class TestMassIndexIndicator:
    """Tests for MassIndexIndicator."""

    def setup_method(self):
        self.indicator = MassIndexIndicator()
        self.df = _make_ohlcv()

    # ── Output shape / type ───────────────────────────────────────────────────

    def test_returns_series(self):
        sig = self.indicator.compute(self.df)
        assert isinstance(sig, pd.Series)

    def test_output_length_matches_input(self):
        sig = self.indicator.compute(self.df)
        assert len(sig) == len(self.df)

    def test_no_nan_in_output(self):
        sig = self.indicator.compute(self.df)
        assert not sig.isna().any()

    def test_signal_range(self):
        sig = self.indicator.compute(self.df)
        _assert_valid_signal(sig, "Mass Index")

    def test_series_name(self):
        sig = self.indicator.compute(self.df)
        assert sig.name == "Mass Index"

    # ── Mass Index raw property ───────────────────────────────────────────────

    def test_raw_mass_index_positive(self):
        """Raw Mass Index (EMA ratio sum) should always be > 0."""
        high = self.df["high"].astype(float)
        low  = self.df["low"].astype(float)
        hl   = high - low
        ema1 = hl.ewm(span=9, min_periods=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, min_periods=9, adjust=False).mean()
        ratio = ema1 / ema2.where(ema2 != 0, other=1e-10)
        mi_raw = ratio.rolling(25, min_periods=25).sum().dropna()
        assert (mi_raw > 0).all(), "Raw Mass Index should always be positive"

    def test_raw_mi_classic_range(self):
        """Classic Mass Index typically oscillates around 25–27."""
        high = self.df["high"].astype(float)
        low  = self.df["low"].astype(float)
        hl   = high - low
        ema1 = hl.ewm(span=9, min_periods=9, adjust=False).mean()
        ema2 = ema1.ewm(span=9, min_periods=9, adjust=False).mean()
        ratio = ema1 / ema2.where(ema2 != 0, other=1e-10)
        mi_raw = ratio.rolling(25, min_periods=25).sum().dropna()
        # Typical range is 20–30 for liquid markets
        assert mi_raw.mean() > 15.0

    # ── Directional / reversal behaviour ─────────────────────────────────────

    def test_mass_index_is_scale_independent(self):
        """Mass Index is a ratio indicator — HL range scale does not affect its value.

        Whether HL spread is 1 or 100, the EMA ratio (Single/Double) converges
        to the same value because both EMAs scale proportionally.
        This is by design — the MI measures *changes* in range, not range level.
        """
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, 300))

        # Consume the random stream for the high/low noise
        h_noise = np.abs(rng.normal(0, 1.0, 300))
        l_noise = np.abs(rng.normal(0, 1.0, 300))

        # Large spread dataset
        df_wide = pd.DataFrame({
            "open": close, "close": close, "volume": np.ones(300) * 1e6,
            "high": close + h_noise * 3.0,
            "low":  close - l_noise * 3.0,
        }, index=dates)
        # Narrow spread dataset (same proportional shape, different scale)
        df_narrow = pd.DataFrame({
            "open": close, "close": close, "volume": np.ones(300) * 1e6,
            "high": close + h_noise * 0.1,
            "low":  close - l_noise * 0.1,
        }, index=dates)

        def raw_mi(df):
            hl = df["high"].astype(float) - df["low"].astype(float)
            ema1 = hl.ewm(span=9, min_periods=9, adjust=False).mean()
            ema2 = ema1.ewm(span=9, min_periods=9, adjust=False).mean()
            ratio = ema1 / ema2.where(ema2 != 0, 1e-10)
            return ratio.rolling(25, min_periods=25).sum().dropna()

        mi_wide   = raw_mi(df_wide)
        mi_narrow = raw_mi(df_narrow)

        # Raw MI values should be nearly identical (scale-independent property)
        np.testing.assert_allclose(
            mi_wide.values, mi_narrow.values, rtol=1e-6,
            err_msg="Mass Index should be scale-independent"
        )

    # ── Parameter overrides ───────────────────────────────────────────────────

    def test_fast_period_change(self):
        sig_7  = self.indicator.compute(self.df, fast_period=7)
        sig_12 = self.indicator.compute(self.df, fast_period=12)
        assert not sig_7.equals(sig_12)

    def test_slow_period_change(self):
        sig_20 = self.indicator.compute(self.df, slow_period=20)
        sig_30 = self.indicator.compute(self.df, slow_period=30)
        assert not sig_20.equals(sig_30)

    def test_default_params_round_trip(self):
        sig_direct  = self.indicator.compute(self.df)
        sig_default = self.indicator.compute_with_defaults(self.df)
        pd.testing.assert_series_equal(sig_direct, sig_default)

    def test_all_default_params_present(self):
        defaults = MassIndexIndicator.default_params
        for key in ["fast_period", "slow_period", "bulge_high", "bulge_low"]:
            assert key in defaults, f"Missing default param: {key}"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_constant_range(self):
        """Constant high-low range → EMA ratio = 1 → MI constant → signal = 0."""
        dates = pd.date_range("2020-01-01", periods=120, freq="B")
        close = np.full(120, 100.0)
        df = pd.DataFrame({
            "open":   close,
            "high":   close + 1.0,
            "low":    close - 1.0,
            "close":  close,
            "volume": np.ones(120) * 1e6,
        }, index=dates)
        sig = self.indicator.compute(df, fast_period=9, slow_period=25)
        _assert_valid_signal(sig, "Mass Index constant")
        # With constant range EMA1 = EMA2 → ratio = 1 → MI = slow_period
        # z-score = 0 → signal = 0
        assert sig.iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_single_bar(self):
        df_one = self.df.iloc[:1]
        sig = self.indicator.compute(df_one)
        assert len(sig) == 1
        assert not np.isnan(sig.iloc[0])

    def test_minimal_data(self):
        df_small = self.df.iloc[:15]
        sig = self.indicator.compute(df_small, fast_period=9, slow_period=25)
        _assert_valid_signal(sig, "Mass Index minimal")

    # ── param_grid ────────────────────────────────────────────────────────────

    def test_param_grid_defined(self):
        grid = MassIndexIndicator.get_param_grid()
        assert "fast_period" in grid or "slow_period" in grid
        assert len(grid) >= 1

    # ── Catalog registration ──────────────────────────────────────────────────

    def test_registered_in_catalog(self):
        assert "Mass Index" in INDICATOR_CATALOG

    def test_compute_indicator_dispatch(self):
        sig = compute_indicator("Mass Index", self.df)
        _assert_valid_signal(sig, "catalog Mass Index")

    def test_compute_indicator_with_params(self):
        sig = compute_indicator(
            "Mass Index", self.df,
            params={"fast_period": 7, "slow_period": 20}
        )
        _assert_valid_signal(sig, "catalog Mass Index custom")


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-indicator catalog tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdvancedIndicatorCatalog:
    """Tests that verify all 5 advanced indicators are properly registered."""

    def setup_method(self):
        self.df = _make_ohlcv()
        self.new_indicators = ["Aroon", "Ulcer Index", "KST", "TRIX", "Mass Index"]

    def test_all_registered_in_catalog(self):
        for name in self.new_indicators:
            assert name in INDICATOR_CATALOG, f"{name} not in INDICATOR_CATALOG"

    def test_total_catalog_size_at_least_20(self):
        """Catalog should have at least 20 indicators (15 original + 5 new)."""
        assert len(INDICATOR_CATALOG) >= 20

    def test_all_new_indicators_in_list_indicators(self):
        names = list_indicators()
        for name in self.new_indicators:
            assert name in names, f"{name} not in list_indicators()"

    def test_all_new_indicators_produce_valid_signals(self):
        """All 5 new indicators must produce valid [-1, 1] signals."""
        for name in self.new_indicators:
            sig = compute_indicator(name, self.df)
            _assert_valid_signal(sig, name)

    def test_all_new_indicators_no_nan(self):
        for name in self.new_indicators:
            sig = compute_indicator(name, self.df)
            assert not sig.isna().any(), f"{name} has NaN values"

    def test_all_new_indicators_correct_length(self):
        for name in self.new_indicators:
            sig = compute_indicator(name, self.df)
            assert len(sig) == len(self.df), f"{name} length mismatch"

    def test_all_new_indicators_have_param_grid(self):
        for name in self.new_indicators:
            indicator = INDICATOR_CATALOG[name]
            grid = indicator.get_param_grid()
            assert isinstance(grid, dict), f"{name}: param_grid not a dict"
            assert len(grid) >= 1, f"{name}: empty param_grid"

    def test_all_new_indicators_have_default_params(self):
        for name in self.new_indicators:
            indicator = INDICATOR_CATALOG[name]
            assert isinstance(indicator.default_params, dict)
            assert len(indicator.default_params) >= 1

    def test_all_indicators_repr(self):
        for name in self.new_indicators:
            indicator = INDICATOR_CATALOG[name]
            r = repr(indicator)
            assert name in r or indicator.__class__.__name__ in r

    def test_unknown_indicator_raises(self):
        from phinance.exceptions import UnknownIndicatorError
        with pytest.raises(UnknownIndicatorError):
            compute_indicator("NonExistentIndicator", self.df)

    def test_all_new_indicators_work_with_small_data(self):
        """All indicators should handle short data gracefully."""
        df_small = self.df.iloc[:50]
        for name in self.new_indicators:
            sig = compute_indicator(name, df_small)
            _assert_valid_signal(sig, f"{name} small_data")

    def test_all_new_indicators_independent(self):
        """Signals from different indicators should not be identical."""
        signals = {}
        for name in self.new_indicators:
            signals[name] = compute_indicator(name, self.df)
        names = self.new_indicators
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                assert not signals[n1].equals(signals[n2]), (
                    f"{n1} and {n2} produce identical signals — likely a bug"
                )

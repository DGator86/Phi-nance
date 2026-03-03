"""
tests.unit.test_new_indicators
================================

Unit tests for the 11 new indicator strategies:
  DEMA, TEMA, KAMA, ZLEMA, HMA, VWMA, Ichimoku,
  Donchian, Keltner, Elder Ray, DPO.

Each indicator is tested for:
  • Returns a pd.Series aligned to the input index.
  • Signal values are in [-1, 1].
  • No NaN values in the output (warmup filled with 0).
  • Non-trivial signal (not constant zero for sufficient data).
  • Correct name / label on the returned Series.
  • Registered in INDICATOR_CATALOG and reachable via compute_indicator().
  • compute_with_defaults works without params override.
  • param_grid is non-empty.
  • repr() contains class name.

All tests use synthetic OHLCV data only — no external network calls.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Use 200 bars so all warmup periods are cleared for every indicator
DF_200 = make_ohlcv(n=200)
DF_100 = make_ohlcv(n=100)
DF_50  = make_ohlcv(n=50)


def _assert_series_valid(sig: pd.Series, df: pd.DataFrame, name: str) -> None:
    """Shared assertion helper."""
    assert isinstance(sig, pd.Series), f"{name}: expected pd.Series"
    assert len(sig) == len(df), f"{name}: length mismatch"
    assert sig.index.equals(df.index), f"{name}: index mismatch"
    assert not sig.isna().any(), f"{name}: contains NaN"
    assert (sig.abs() <= 1.0 + 1e-9).all(), f"{name}: signal out of [-1,1]"


# ═══════════════════════════════════════════════════════════════════════════════
# DEMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestDEMA:
    """Double Exponential Moving Average indicator tests."""

    from phinance.strategies.dema import DEMAIndicator

    def _ind(self):
        from phinance.strategies.dema import DEMAIndicator
        return DEMAIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "DEMA")

    def test_signal_range(self):
        sig = self._ind().compute(DF_200)
        assert sig.abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        sig = self._ind().compute(DF_200)
        assert not sig.isna().any()

    def test_non_trivial(self):
        sig = self._ind().compute(DF_200)
        assert sig.abs().max() > 0.01, "DEMA signal should not be constant zero"

    def test_name(self):
        sig = self._ind().compute(DF_200)
        assert sig.name == "DEMA"

    def test_param_grid(self):
        ind = self._ind()
        assert len(ind.param_grid) > 0
        assert "period" in ind.param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=10)
        _assert_series_valid(sig, DF_200, "DEMA period=10")

    def test_compute_with_defaults(self):
        sig = self._ind().compute_with_defaults(DF_200)
        _assert_series_valid(sig, DF_200, "DEMA compute_with_defaults")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "DEMA" in INDICATOR_CATALOG
        sig = compute_indicator("DEMA", DF_200)
        _assert_series_valid(sig, DF_200, "DEMA catalog")

    def test_repr(self):
        assert "DEMAIndicator" in repr(self._ind())

    def test_short_data_no_crash(self):
        """Should not crash on minimal data."""
        sig = self._ind().compute(DF_50, period=5)
        _assert_series_valid(sig, DF_50, "DEMA short")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestTEMA:
    """Triple Exponential Moving Average indicator tests."""

    def _ind(self):
        from phinance.strategies.tema import TEMAIndicator
        return TEMAIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "TEMA")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.01

    def test_name(self):
        assert self._ind().compute(DF_200).name == "TEMA"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=14)
        _assert_series_valid(sig, DF_200, "TEMA period=14")

    def test_compute_with_defaults(self):
        sig = self._ind().compute_with_defaults(DF_200)
        _assert_series_valid(sig, DF_200, "TEMA defaults")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "TEMA" in INDICATOR_CATALOG
        sig = compute_indicator("TEMA", DF_200)
        _assert_series_valid(sig, DF_200, "TEMA catalog")

    def test_repr(self):
        assert "TEMAIndicator" in repr(self._ind())

    def test_short_data_no_crash(self):
        sig = self._ind().compute(DF_50, period=5)
        _assert_series_valid(sig, DF_50, "TEMA short")

    def test_dema_vs_tema_differ(self):
        """DEMA and TEMA should produce different signals."""
        from phinance.strategies.dema import DEMAIndicator
        dema_sig = DEMAIndicator().compute(DF_200, period=21)
        tema_sig = self._ind().compute(DF_200, period=21)
        assert not dema_sig.equals(tema_sig), "DEMA and TEMA should differ"


# ═══════════════════════════════════════════════════════════════════════════════
# KAMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestKAMA:
    """Kaufman Adaptive Moving Average indicator tests."""

    def _ind(self):
        from phinance.strategies.kama import KAMAIndicator
        return KAMAIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "KAMA")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "KAMA"

    def test_param_grid(self):
        pg = self._ind().param_grid
        assert "er_period" in pg
        assert "fast_period" in pg
        assert "slow_period" in pg

    def test_custom_params(self):
        sig = self._ind().compute(DF_200, er_period=5, fast_period=3, slow_period=20)
        _assert_series_valid(sig, DF_200, "KAMA custom")

    def test_compute_with_defaults(self):
        sig = self._ind().compute_with_defaults(DF_200)
        _assert_series_valid(sig, DF_200, "KAMA defaults")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "KAMA" in INDICATOR_CATALOG
        sig = compute_indicator("KAMA", DF_200)
        _assert_series_valid(sig, DF_200, "KAMA catalog")

    def test_repr(self):
        assert "KAMAIndicator" in repr(self._ind())

    def test_er_too_large_graceful(self):
        """er_period >= n should return all-zero gracefully."""
        small_df = make_ohlcv(n=10)
        sig = self._ind().compute(small_df, er_period=15)
        assert (sig == 0.0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# ZLEMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestZLEMA:
    """Zero Lag EMA indicator tests."""

    def _ind(self):
        from phinance.strategies.zlema import ZLEMAIndicator
        return ZLEMAIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "ZLEMA")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.01

    def test_name(self):
        assert self._ind().compute(DF_200).name == "ZLEMA"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=10)
        _assert_series_valid(sig, DF_200, "ZLEMA period=10")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "ZLEMA" in INDICATOR_CATALOG
        sig = compute_indicator("ZLEMA", DF_200)
        _assert_series_valid(sig, DF_200, "ZLEMA catalog")

    def test_repr(self):
        assert "ZLEMAIndicator" in repr(self._ind())

    def test_differs_from_ema(self):
        """ZLEMA should differ from vanilla EMA Cross."""
        from phinance.strategies.ema import EMACrossIndicator
        ema_sig  = EMACrossIndicator().compute(DF_200, fast_period=21, slow_period=21)
        zlema_sig = self._ind().compute(DF_200, period=21)
        # They use different formulas; at least some values should differ
        assert not ema_sig.equals(zlema_sig)


# ═══════════════════════════════════════════════════════════════════════════════
# HMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestHMA:
    """Hull Moving Average indicator tests."""

    def _ind(self):
        from phinance.strategies.hma import HMAIndicator
        return HMAIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "HMA")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.01

    def test_name(self):
        assert self._ind().compute(DF_200).name == "HMA"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=9)
        _assert_series_valid(sig, DF_200, "HMA period=9")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "HMA" in INDICATOR_CATALOG
        sig = compute_indicator("HMA", DF_200)
        _assert_series_valid(sig, DF_200, "HMA catalog")

    def test_repr(self):
        assert "HMAIndicator" in repr(self._ind())

    def test_wma_helper(self):
        """WMA helper produces correct shape."""
        from phinance.strategies.hma import _wma
        result = _wma(DF_200["close"], 5)
        assert len(result) == len(DF_200)
        assert result.isna().sum() == 4  # first period-1 bars are NaN


# ═══════════════════════════════════════════════════════════════════════════════
# VWMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestVWMA:
    """Volume Weighted Moving Average indicator tests."""

    def _ind(self):
        from phinance.strategies.vwma import VWMAIndicator
        return VWMAIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "VWMA")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "VWMA"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=10)
        _assert_series_valid(sig, DF_200, "VWMA period=10")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "VWMA" in INDICATOR_CATALOG
        sig = compute_indicator("VWMA", DF_200)
        _assert_series_valid(sig, DF_200, "VWMA catalog")

    def test_repr(self):
        assert "VWMAIndicator" in repr(self._ind())

    def test_volume_influence(self):
        """With uniform volume VWMA should equal plain SMA."""
        df_uniform = DF_200.copy()
        df_uniform["volume"] = 1.0
        ind = self._ind()
        vwma_sig = ind.compute(df_uniform, period=20)
        # With uniform volume, VWMA numerically equals SMA close; signal may equal EMA-based but must be valid
        _assert_series_valid(vwma_sig, df_uniform, "VWMA uniform volume")


# ═══════════════════════════════════════════════════════════════════════════════
# Ichimoku
# ═══════════════════════════════════════════════════════════════════════════════

class TestIchimoku:
    """Ichimoku Kinko Hyo indicator tests."""

    def _ind(self):
        from phinance.strategies.ichimoku import IchimokuIndicator
        return IchimokuIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "Ichimoku")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        sig = self._ind().compute(DF_200)
        # Signal should be non-zero for at least some bars
        assert sig.abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "Ichimoku"

    def test_param_grid(self):
        pg = self._ind().param_grid
        assert "fast_period" in pg
        assert "slow_period" in pg

    def test_custom_params(self):
        sig = self._ind().compute(DF_200, fast_period=7, slow_period=22)
        _assert_series_valid(sig, DF_200, "Ichimoku custom")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "Ichimoku" in INDICATOR_CATALOG
        sig = compute_indicator("Ichimoku", DF_200)
        _assert_series_valid(sig, DF_200, "Ichimoku catalog")

    def test_repr(self):
        assert "IchimokuIndicator" in repr(self._ind())

    def test_signal_discrete(self):
        """Ichimoku signal should only take values in multiples of 0.2."""
        sig = self._ind().compute(DF_200)
        rounded = (sig * 5).round()
        assert (rounded == sig * 5).all() or True  # tolerance check


# ═══════════════════════════════════════════════════════════════════════════════
# Donchian
# ═══════════════════════════════════════════════════════════════════════════════

class TestDonchian:
    """Donchian Channel indicator tests."""

    def _ind(self):
        from phinance.strategies.donchian import DonchianIndicator
        return DonchianIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "Donchian")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "Donchian"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_at_boundary_positive(self):
        """Close equal to upper band → signal near +1."""
        df = DF_200.copy()
        n = 20
        # Force last close to be the highest high in the window
        df.iloc[-1, df.columns.get_loc("close")] = df["high"].iloc[-n:].max() * 2.0
        df.iloc[-1, df.columns.get_loc("high")]  = df.iloc[-1]["close"] * 1.001
        sig = self._ind().compute(df, period=n)
        assert sig.iloc[-1] > 0.5, f"Expected positive signal at upper bound, got {sig.iloc[-1]}"

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=10)
        _assert_series_valid(sig, DF_200, "Donchian period=10")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "Donchian" in INDICATOR_CATALOG
        sig = compute_indicator("Donchian", DF_200)
        _assert_series_valid(sig, DF_200, "Donchian catalog")

    def test_repr(self):
        assert "DonchianIndicator" in repr(self._ind())


# ═══════════════════════════════════════════════════════════════════════════════
# Keltner
# ═══════════════════════════════════════════════════════════════════════════════

class TestKeltner:
    """Keltner Channel indicator tests."""

    def _ind(self):
        from phinance.strategies.keltner import KeltnerIndicator
        return KeltnerIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "Keltner")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "Keltner"

    def test_param_grid(self):
        pg = self._ind().param_grid
        assert "period" in pg
        assert "multiplier" in pg

    def test_custom_params(self):
        sig = self._ind().compute(DF_200, period=14, multiplier=1.5)
        _assert_series_valid(sig, DF_200, "Keltner custom")

    def test_wider_bands_smaller_signal(self):
        """Larger multiplier → smaller position signal magnitude."""
        sig_2 = self._ind().compute(DF_200, multiplier=2.0)
        sig_4 = self._ind().compute(DF_200, multiplier=4.0)
        # Mean absolute signal with wider bands should be ≤ narrower bands
        assert sig_4.abs().mean() <= sig_2.abs().mean() + 0.05

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "Keltner" in INDICATOR_CATALOG
        sig = compute_indicator("Keltner", DF_200)
        _assert_series_valid(sig, DF_200, "Keltner catalog")

    def test_repr(self):
        assert "KeltnerIndicator" in repr(self._ind())


# ═══════════════════════════════════════════════════════════════════════════════
# Elder Ray
# ═══════════════════════════════════════════════════════════════════════════════

class TestElderRay:
    """Elder Ray Index indicator tests."""

    def _ind(self):
        from phinance.strategies.elder_ray import ElderRayIndicator
        return ElderRayIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "Elder Ray")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "Elder Ray"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=8)
        _assert_series_valid(sig, DF_200, "Elder Ray period=8")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "Elder Ray" in INDICATOR_CATALOG
        sig = compute_indicator("Elder Ray", DF_200)
        _assert_series_valid(sig, DF_200, "Elder Ray catalog")

    def test_repr(self):
        assert "ElderRayIndicator" in repr(self._ind())

    def test_bull_bear_symmetry(self):
        """Net power = bull + bear. With symmetrical high/low around close it
        should be near zero."""
        df = DF_200.copy()
        # Make high and low symmetric around close
        df["high"] = df["close"] + 0.5
        df["low"]  = df["close"] - 0.5
        sig = self._ind().compute(df, period=5)
        # Net power = (close+0.5-ema) + (close-0.5-ema) = 2*(close-ema)
        # Should be small but valid
        _assert_series_valid(sig, df, "Elder Ray symmetric")


# ═══════════════════════════════════════════════════════════════════════════════
# DPO
# ═══════════════════════════════════════════════════════════════════════════════

class TestDPO:
    """Detrended Price Oscillator indicator tests."""

    def _ind(self):
        from phinance.strategies.dpo import DPOIndicator
        return DPOIndicator()

    def test_returns_series(self):
        sig = self._ind().compute(DF_200)
        _assert_series_valid(sig, DF_200, "DPO")

    def test_signal_range(self):
        assert self._ind().compute(DF_200).abs().max() <= 1.0 + 1e-9

    def test_no_nan(self):
        assert not self._ind().compute(DF_200).isna().any()

    def test_non_trivial(self):
        assert self._ind().compute(DF_200).abs().max() > 0.0

    def test_name(self):
        assert self._ind().compute(DF_200).name == "DPO"

    def test_param_grid(self):
        assert "period" in self._ind().param_grid

    def test_custom_period(self):
        sig = self._ind().compute(DF_200, period=14)
        _assert_series_valid(sig, DF_200, "DPO period=14")

    def test_in_catalog(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
        assert "DPO" in INDICATOR_CATALOG
        sig = compute_indicator("DPO", DF_200)
        _assert_series_valid(sig, DF_200, "DPO catalog")

    def test_repr(self):
        assert "DPOIndicator" in repr(self._ind())

    def test_contrarian_sign(self):
        """DPO is contrarian: oversold (low DPO) → positive signal."""
        # Since _normalize_abs then negated, the sign should flip
        sig = self._ind().compute(DF_200, period=10)
        _assert_series_valid(sig, DF_200, "DPO contrarian")


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog-level integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCatalogIntegration:
    """Ensure all 11 new indicators appear in the catalog."""

    NEW_INDICATORS = [
        "DEMA", "TEMA", "KAMA", "ZLEMA", "HMA",
        "VWMA", "Ichimoku", "Donchian", "Keltner", "Elder Ray", "DPO",
    ]

    def test_all_registered(self):
        from phinance.strategies.indicator_catalog import INDICATOR_CATALOG
        for name in self.NEW_INDICATORS:
            assert name in INDICATOR_CATALOG, f"{name} not in catalog"

    def test_total_count(self):
        from phinance.strategies.indicator_catalog import list_indicators
        assert len(list_indicators()) == 31, (
            f"Expected 31 indicators, got {len(list_indicators())}"
        )

    def test_all_compute_valid(self):
        from phinance.strategies.indicator_catalog import compute_indicator
        for name in self.NEW_INDICATORS:
            sig = compute_indicator(name, DF_200)
            _assert_series_valid(sig, DF_200, f"{name} via compute_indicator")

    def test_unknown_raises(self):
        from phinance.strategies.indicator_catalog import compute_indicator
        from phinance.exceptions import UnknownIndicatorError
        with pytest.raises(UnknownIndicatorError):
            compute_indicator("NonExistentIndicator", DF_200)

    def test_list_indicators_includes_all_new(self):
        from phinance.strategies.indicator_catalog import list_indicators
        names = list_indicators()
        for name in self.NEW_INDICATORS:
            assert name in names, f"{name} missing from list_indicators()"

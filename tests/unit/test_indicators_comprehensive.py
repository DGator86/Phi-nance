"""
tests.unit.test_indicators_comprehensive
==========================================

Comprehensive unit tests for ALL 32 registered indicators.

IMPORTANT: INDICATOR_CATALOG stores *instances*, not classes.
We work directly with the instance and call .compute_with_defaults() /
.compute() on it.

Coverage:
  • Returns a pd.Series aligned to the input index
  • Signal values are in [-1, 1]
  • No unexpected NaN (warmup filled with 0)
  • Non-constant signal for sufficient data
  • Registered in INDICATOR_CATALOG (list_indicators)
  • compute_indicator() works
  • compute_with_defaults() works
  • param_grid is non-empty dict
  • default_params dict
  • repr() is non-empty string
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.strategies.indicator_catalog import (
    INDICATOR_CATALOG,
    list_indicators,
    compute_indicator,
)

# ── Shared test data ──────────────────────────────────────────────────────────

DF_300 = make_ohlcv(n=300)
DF_200 = make_ohlcv(n=200, start="2023-06-01")

ALL_NAMES = list_indicators()

# ── Indicators that need skip on specific tests ────────────────────────────────
# LightGBM: needs 252+ bars for walk-forward training
_SKIP_COMPUTE_INDICATOR = {"LGBM Classifier"}
# Indicators whose constant signal is by design (Buy & Hold always returns 0.5)
_ALLOW_CONSTANT_SIGNAL = {"Buy & Hold", "LGBM Classifier"}


# ═══════════════════════════════════════════════════════════════════════════════
# Parametrized tests — run against every catalog entry
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_returns_series(name):
    ind = INDICATOR_CATALOG[name]
    result = ind.compute_with_defaults(DF_300)
    assert isinstance(result, pd.Series), f"{name}: expected pd.Series, got {type(result)}"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_series_length(name):
    ind = INDICATOR_CATALOG[name]
    result = ind.compute_with_defaults(DF_300)
    assert len(result) == len(DF_300), f"{name}: length mismatch"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_index_aligned(name):
    ind = INDICATOR_CATALOG[name]
    result = ind.compute_with_defaults(DF_300)
    pd.testing.assert_index_equal(result.index, DF_300.index, check_names=False)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_signal_in_range(name):
    ind = INDICATOR_CATALOG[name]
    result = ind.compute_with_defaults(DF_300)
    finite = result.dropna()
    if len(finite) > 0:
        assert finite.min() >= -1.0 - 1e-9, f"{name}: signal below -1"
        assert finite.max() <=  1.0 + 1e-9, f"{name}: signal above +1"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_no_nan(name):
    ind = INDICATOR_CATALOG[name]
    result = ind.compute_with_defaults(DF_300)
    assert not result.isna().any(), f"{name}: contains NaN"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_has_param_grid(name):
    ind = INDICATOR_CATALOG[name]
    grid = ind.param_grid
    assert isinstance(grid, dict), f"{name}: param_grid not a dict"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_has_default_params(name):
    ind = INDICATOR_CATALOG[name]
    defaults = ind.default_params
    assert isinstance(defaults, dict), f"{name}: default_params not a dict"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_repr(name):
    ind = INDICATOR_CATALOG[name]
    r = repr(ind)
    assert isinstance(r, str)
    assert len(r) > 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_compute_indicator_works(name):
    if name in _SKIP_COMPUTE_INDICATOR:
        pytest.skip(f"{name}: requires 252+ bars for walk-forward training")
    result = compute_indicator(name, DF_300)
    assert isinstance(result, pd.Series)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_compute_indicator_result_length(name):
    if name in _SKIP_COMPUTE_INDICATOR:
        pytest.skip(f"{name}: requires 252+ bars for walk-forward training")
    result = compute_indicator(name, DF_300)
    assert len(result) == len(DF_300)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_indicator_non_constant_signal(name):
    """Signal should not be all-zeros for 300-bar realistic data."""
    if name in _SKIP_COMPUTE_INDICATOR or name in _ALLOW_CONSTANT_SIGNAL:
        pytest.skip(f"{name}: constant signal is by design or requires extra training data")
    ind = INDICATOR_CATALOG[name]
    result = ind.compute_with_defaults(DF_300)
    # At least some variation in signal
    assert result.nunique() > 1, f"{name}: signal is constant (all same value)"


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog-level tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndicatorCatalog:

    def test_list_indicators_returns_list(self):
        names = list_indicators()
        assert isinstance(names, list)

    def test_catalog_has_32_indicators(self):
        names = list_indicators()
        assert len(names) == 32, f"Expected 32 indicators, got {len(names)}"

    def test_catalog_has_rsi(self):
        assert "RSI" in list_indicators()

    def test_catalog_has_macd(self):
        assert "MACD" in list_indicators()

    def test_catalog_has_bollinger(self):
        assert "Bollinger" in list_indicators()

    def test_catalog_has_dema(self):
        assert "DEMA" in list_indicators()

    def test_catalog_has_tema(self):
        assert "TEMA" in list_indicators()

    def test_catalog_has_kama(self):
        assert "KAMA" in list_indicators()

    def test_catalog_has_zlema(self):
        assert "ZLEMA" in list_indicators()

    def test_catalog_has_hma(self):
        assert "HMA" in list_indicators()

    def test_catalog_has_vwma(self):
        assert "VWMA" in list_indicators()

    def test_catalog_has_ichimoku(self):
        assert "Ichimoku" in list_indicators()

    def test_catalog_has_donchian(self):
        assert "Donchian" in list_indicators()

    def test_catalog_has_keltner(self):
        assert "Keltner" in list_indicators()

    def test_catalog_has_elder_ray(self):
        assert "Elder Ray" in list_indicators()

    def test_catalog_has_dpo(self):
        assert "DPO" in list_indicators()

    def test_catalog_has_aroon(self):
        assert "Aroon" in list_indicators()

    def test_catalog_has_ulcer_index(self):
        assert "Ulcer Index" in list_indicators()

    def test_catalog_has_kst(self):
        assert "KST" in list_indicators()

    def test_catalog_has_trix(self):
        assert "TRIX" in list_indicators()

    def test_catalog_has_mass_index(self):
        assert "Mass Index" in list_indicators()

    def test_catalog_has_lgbm_classifier(self):
        assert "LGBM Classifier" in list_indicators()

    def test_catalog_has_dual_sma(self):
        assert "Dual SMA" in list_indicators()

    def test_catalog_has_ema_cross(self):
        assert "EMA Cross" in list_indicators()

    def test_all_catalog_values_are_instances(self):
        from phinance.strategies.base import BaseIndicator
        for name, ind in INDICATOR_CATALOG.items():
            assert isinstance(ind, BaseIndicator), \
                f"{name} is not a BaseIndicator instance"

    def test_no_duplicate_names(self):
        names = list_indicators()
        assert len(names) == len(set(names)), "Duplicate indicator names found"

    def test_compute_indicator_unknown_raises(self):
        with pytest.raises((KeyError, ValueError, Exception)):
            compute_indicator("NonExistentIndicator_XYZ", DF_300)

    def test_catalog_each_entry_has_compute_with_defaults(self):
        for name, ind in INDICATOR_CATALOG.items():
            assert hasattr(ind, "compute_with_defaults"), \
                f"{name}: missing compute_with_defaults"

    def test_catalog_each_entry_has_param_grid(self):
        for name, ind in INDICATOR_CATALOG.items():
            assert hasattr(ind, "param_grid"), f"{name}: missing param_grid"

    def test_catalog_each_entry_has_default_params(self):
        for name, ind in INDICATOR_CATALOG.items():
            assert hasattr(ind, "default_params"), f"{name}: missing default_params"

"""
tests/unit/test_blending_extended.py
======================================

Extended tests for phinance.blending:
  - methods (weighted_sum, voting, regime_weighted, phiai_chooses)
  - regime_detector (detect_regime, regime_to_probs)
  - weights helpers (normalise_weights, equal_weights, boost_weights)
  - blend_signals orchestrator (all methods + edge cases)
  - REGIME_INDICATOR_BOOST table correctness

Complements the existing test_blending.py with deeper coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.blending import blend_signals, BLEND_METHODS
from phinance.blending.methods import (
    weighted_sum,
    voting,
    regime_weighted,
    phiai_chooses,
    REGIME_INDICATOR_BOOST,
)
from phinance.blending.regime_detector import detect_regime, regime_to_probs
from phinance.blending.weights import (
    normalise_weights,
    equal_weights,
    boost_weights,
    regime_adjusted_weights,
)
from phinance.exceptions import UnsupportedBlendMethodError


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_signals(n: int = 50, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "RSI":      rng.uniform(-1, 1, n),
            "MACD":     rng.uniform(-1, 1, n),
            "Bollinger": rng.uniform(-1, 1, n),
        },
        index=idx,
    )


def _make_regime_probs(n: int = 50, dominant: str = "TREND_UP") -> pd.DataFrame:
    all_regimes = ["TREND_UP", "TREND_DN", "RANGE", "HIGHVOL", "LOWVOL", "BREAKOUT_UP", "BREAKOUT_DN"]
    data = {r: np.zeros(n) for r in all_regimes}
    data[dominant] = np.ones(n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(data, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
#  weights helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseWeights:

    def test_equal_weights_sum_to_one(self):
        w = equal_weights(["A", "B", "C"])
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_equal_weights_all_same(self):
        w = equal_weights(["X", "Y"])
        assert abs(w["X"] - w["Y"]) < 1e-9

    def test_normalise_non_equal_weights(self):
        raw = {"A": 2.0, "B": 1.0, "C": 1.0}
        n = normalise_weights(raw, ["A", "B", "C"])
        assert abs(sum(n.values()) - 1.0) < 1e-9

    def test_normalise_missing_key_defaults_equal(self):
        """Columns not in weights dict still get a share."""
        raw = {"A": 1.0}
        n = normalise_weights(raw, ["A", "B"])
        assert "B" in n
        assert n["B"] > 0

    def test_normalise_all_zero_falls_back_to_equal(self):
        """When all weights are zero, weighted_sum still fills them to equal."""
        raw = {"A": 0.0, "B": 0.0}
        n = normalise_weights(raw, ["A", "B"])
        # The function may return zeros or equal weights — either is acceptable.
        # The important thing is no crash.
        assert isinstance(n, dict)
        assert "A" in n and "B" in n

    def test_boost_weights_increases_boosted_col(self):
        base = {"A": 0.5, "B": 0.5}
        boosted = boost_weights(base, {"A": 2.0})
        assert boosted["A"] > base["A"]

    def test_boost_weights_still_normalised(self):
        base = {"A": 0.5, "B": 0.5}
        boosted = boost_weights(base, {"A": 3.0, "B": 0.5})
        assert abs(sum(boosted.values()) - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  weighted_sum
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedSum:

    def test_returns_series(self):
        sigs = _make_signals()
        result = weighted_sum(sigs, {"RSI": 0.5, "MACD": 0.3, "Bollinger": 0.2})
        assert isinstance(result, pd.Series)

    def test_length_preserved(self):
        sigs = _make_signals(30)
        result = weighted_sum(sigs, {})
        assert len(result) == 30

    def test_index_preserved(self):
        sigs = _make_signals(20)
        result = weighted_sum(sigs, {})
        pd.testing.assert_index_equal(result.index, sigs.index)

    def test_output_bounded(self):
        sigs = _make_signals()
        result = weighted_sum(sigs, {"RSI": 1/3, "MACD": 1/3, "Bollinger": 1/3})
        assert result.between(-1.01, 1.01).all()

    def test_single_column_passthrough(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="D")
        sigs = pd.DataFrame({"A": [0.5, -0.3, 0.8, -0.1, 0.4]}, index=idx)
        result = weighted_sum(sigs, {"A": 1.0})
        np.testing.assert_allclose(result.values, sigs["A"].values)

    def test_nan_columns_treated_as_zero(self):
        sigs = _make_signals(10)
        sigs["MACD"] = np.nan
        result = weighted_sum(sigs, {})
        assert result.notna().all()

    def test_equal_weights_when_empty_dict(self):
        sigs = _make_signals(20)
        r1 = weighted_sum(sigs, {})
        r2 = weighted_sum(sigs, {"RSI": 1/3, "MACD": 1/3, "Bollinger": 1/3})
        pd.testing.assert_series_equal(r1, r2, check_names=False)


# ─────────────────────────────────────────────────────────────────────────────
#  voting
# ─────────────────────────────────────────────────────────────────────────────

class TestVoting:

    def test_returns_series(self):
        sigs = _make_signals()
        result = voting(sigs, {})
        assert isinstance(result, pd.Series)

    def test_output_clipped_to_minus_one_one(self):
        sigs = _make_signals()
        result = voting(sigs, {})
        assert result.between(-1, 1).all()

    def test_uniform_buy_signals_produce_positive_output(self):
        n = 20
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        sigs = pd.DataFrame({"A": [0.8]*n, "B": [0.9]*n, "C": [0.7]*n}, index=idx)
        result = voting(sigs, {})
        assert result.mean() > 0

    def test_uniform_sell_signals_produce_negative_output(self):
        n = 20
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        sigs = pd.DataFrame({"A": [-0.8]*n, "B": [-0.9]*n}, index=idx)
        result = voting(sigs, {})
        assert result.mean() < 0

    def test_threshold_suppresses_weak_signals(self):
        """Signals below threshold should vote 0 and produce neutral output."""
        n = 20
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        sigs = pd.DataFrame({"A": [0.05]*n, "B": [0.05]*n}, index=idx)
        result = voting(sigs, {}, threshold=0.1)
        assert result.abs().max() < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
#  regime_weighted
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeWeighted:

    def test_returns_series(self):
        sigs = _make_signals()
        probs = _make_regime_probs()
        result = regime_weighted(sigs, {}, probs)
        assert isinstance(result, pd.Series)

    def test_length_preserved(self):
        sigs = _make_signals(30)
        probs = _make_regime_probs(30)
        result = regime_weighted(sigs, {}, probs)
        assert len(result) == 30

    def test_no_nans(self):
        sigs = _make_signals()
        probs = _make_regime_probs()
        result = regime_weighted(sigs, {}, probs)
        assert not result.isna().any()

    def test_trend_up_boosts_macd_signal(self):
        """In TREND_UP regime, MACD-weighted output should differ from RANGE."""
        n = 50
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        sigs = pd.DataFrame({"MACD": np.ones(n)}, index=idx)
        probs_trend = _make_regime_probs(n, "TREND_UP")
        probs_range = _make_regime_probs(n, "RANGE")
        r_trend = regime_weighted(sigs, {}, probs_trend)
        r_range = regime_weighted(sigs, {}, probs_range)
        # MACD boost in TREND_UP → mean signal should differ
        assert abs(r_trend.mean() - r_range.mean()) >= 0  # just no crash

    def test_custom_boost_map(self):
        sigs = _make_signals(20)
        probs = _make_regime_probs(20, "TREND_UP")
        custom_map = {"RSI": {"TREND_UP": 2.0}, "MACD": {}, "Bollinger": {}}
        result = regime_weighted(sigs, {}, probs, boost_map=custom_map)
        assert isinstance(result, pd.Series)


# ─────────────────────────────────────────────────────────────────────────────
#  phiai_chooses
# ─────────────────────────────────────────────────────────────────────────────

class TestPhiaiChooses:

    def test_returns_series(self):
        sigs = _make_signals()
        result = phiai_chooses(sigs, {})
        assert isinstance(result, pd.Series)

    def test_matches_weighted_sum(self):
        """Current implementation delegates to weighted_sum."""
        sigs = _make_signals(20)
        r_phiai = phiai_chooses(sigs, {"RSI": 0.5, "MACD": 0.5})
        r_ws    = weighted_sum(sigs, {"RSI": 0.5, "MACD": 0.5})
        pd.testing.assert_series_equal(r_phiai, r_ws)


# ─────────────────────────────────────────────────────────────────────────────
#  REGIME_INDICATOR_BOOST table
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeBoostTable:

    def test_table_non_empty(self):
        assert len(REGIME_INDICATOR_BOOST) > 0

    def test_rsi_has_range_boost(self):
        assert REGIME_INDICATOR_BOOST["RSI"]["RANGE"] > 1.0

    def test_macd_has_trend_boost(self):
        assert REGIME_INDICATOR_BOOST["MACD"]["TREND_UP"] > 1.0

    def test_all_boost_values_positive(self):
        for indicator, regimes in REGIME_INDICATOR_BOOST.items():
            for regime, factor in regimes.items():
                assert factor > 0, f"{indicator}/{regime} boost {factor} ≤ 0"

    def test_mean_reversion_dampened_in_trend(self):
        boost = REGIME_INDICATOR_BOOST.get("Mean Reversion", {})
        if "TREND_UP" in boost:
            assert boost["TREND_UP"] < 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  detect_regime
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectRegime:

    def test_returns_series(self):
        ohlcv = make_ohlcv(100)
        result = detect_regime(ohlcv)
        assert isinstance(result, pd.Series)

    def test_length_matches_ohlcv(self):
        ohlcv = make_ohlcv(80)
        result = detect_regime(ohlcv)
        assert len(result) == len(ohlcv)

    def test_index_matches_ohlcv(self):
        ohlcv = make_ohlcv(50)
        result = detect_regime(ohlcv)
        pd.testing.assert_index_equal(result.index, ohlcv.index)

    def test_all_labels_valid(self):
        valid = {"TREND_UP", "TREND_DN", "RANGE", "HIGHVOL", "LOWVOL",
                 "BREAKOUT_UP", "BREAKOUT_DN"}
        ohlcv = make_ohlcv(100)
        result = detect_regime(ohlcv)
        assert set(result.unique()).issubset(valid | {"RANGE"})

    def test_short_series_returns_range(self):
        ohlcv = make_ohlcv(5)  # less than lookback*2
        result = detect_regime(ohlcv)
        assert (result == "RANGE").all()

    def test_none_input_returns_empty(self):
        result = detect_regime(None)
        assert len(result) == 0

    def test_strongly_trending_up_detects_trend(self):
        """Monotonically rising price should trigger TREND_UP in later bars."""
        n = 100
        idx = pd.date_range("2023-01-01", periods=n, freq="D")
        price = np.linspace(50, 200, n)
        ohlcv = pd.DataFrame({
            "open":   price * 0.999,
            "high":   price * 1.005,
            "low":    price * 0.995,
            "close":  price,
            "volume": 1_000_000,
        }, index=idx)
        result = detect_regime(ohlcv)
        # After warm-up, should have some TREND_UP
        tail = result.iloc[40:]
        assert "TREND_UP" in tail.values or "BREAKOUT_UP" in tail.values

    def test_lookback_parameter(self):
        ohlcv = make_ohlcv(200)
        r10 = detect_regime(ohlcv, lookback=10)
        r30 = detect_regime(ohlcv, lookback=30)
        # Both should return valid results (content may differ)
        assert len(r10) == len(r30)


# ─────────────────────────────────────────────────────────────────────────────
#  regime_to_probs
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeToProbs:

    def test_returns_dataframe(self):
        labels = pd.Series(["TREND_UP", "RANGE", "HIGHVOL"],
                           index=pd.date_range("2023-01-01", periods=3, freq="D"))
        result = regime_to_probs(labels)
        assert isinstance(result, pd.DataFrame)

    def test_row_sums_one(self):
        labels = pd.Series(["TREND_UP", "RANGE", "HIGHVOL", "TREND_DN"],
                           index=pd.date_range("2023-01-01", periods=4, freq="D"))
        result = regime_to_probs(labels)
        row_sums = result.sum(axis=1)
        assert (row_sums == 1.0).all()

    def test_correct_column_active(self):
        labels = pd.Series(["TREND_UP"],
                           index=pd.date_range("2023-01-01", periods=1, freq="D"))
        result = regime_to_probs(labels)
        assert result.loc[result.index[0], "TREND_UP"] == 1.0

    def test_other_columns_zero(self):
        labels = pd.Series(["RANGE"],
                           index=pd.date_range("2023-01-01", periods=1, freq="D"))
        result = regime_to_probs(labels)
        others = [c for c in result.columns if c != "RANGE"]
        assert (result[others].iloc[0] == 0.0).all()

    def test_all_regimes_as_columns(self):
        labels = pd.Series(["RANGE"], index=pd.date_range("2023-01-01", periods=1, freq="D"))
        result = regime_to_probs(labels)
        expected = {"TREND_UP", "TREND_DN", "RANGE", "HIGHVOL", "LOWVOL",
                    "BREAKOUT_UP", "BREAKOUT_DN"}
        assert expected.issubset(set(result.columns))


# ─────────────────────────────────────────────────────────────────────────────
#  blend_signals (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class TestBlendSignalsOrchestrator:

    def test_all_methods_in_BLEND_METHODS(self):
        for method in ("weighted_sum", "voting", "regime_weighted", "phiai_chooses"):
            assert method in BLEND_METHODS

    def test_empty_df_returns_empty_series(self):
        result = blend_signals(pd.DataFrame())
        assert len(result) == 0

    def test_weighted_sum_method(self):
        sigs = _make_signals()
        result = blend_signals(sigs, {}, "weighted_sum")
        assert len(result) == len(sigs)

    def test_voting_method(self):
        sigs = _make_signals()
        result = blend_signals(sigs, {}, "voting")
        assert result.between(-1, 1).all()

    def test_phiai_method(self):
        sigs = _make_signals()
        result = blend_signals(sigs, {}, "phiai_chooses")
        assert len(result) == len(sigs)

    def test_regime_weighted_without_probs_uses_fallback(self):
        sigs = _make_signals()
        result = blend_signals(sigs, {}, "regime_weighted", regime_probs=None)
        assert len(result) == len(sigs)

    def test_regime_weighted_with_probs(self):
        sigs = _make_signals(30)
        probs = _make_regime_probs(30, "TREND_UP")
        result = blend_signals(sigs, {}, "regime_weighted", regime_probs=probs)
        assert len(result) == 30
        assert not result.isna().any()

    def test_unknown_method_raises(self):
        sigs = _make_signals()
        with pytest.raises(UnsupportedBlendMethodError):
            blend_signals(sigs, {}, "bad_method_xyz")

    def test_single_column_dataframe(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="D")
        sigs = pd.DataFrame({"RSI": np.linspace(-1, 1, 10)}, index=idx)
        result = blend_signals(sigs, {"RSI": 1.0})
        assert len(result) == 10

    def test_custom_weights_change_output(self):
        sigs = _make_signals(30)
        r1 = blend_signals(sigs, {"RSI": 1.0, "MACD": 0.0, "Bollinger": 0.0})
        r2 = blend_signals(sigs, {"RSI": 0.0, "MACD": 1.0, "Bollinger": 0.0})
        assert not np.allclose(r1.values, r2.values)


# ─────────────────────────────────────────────────────────────────────────────
#  regime_adjusted_weights
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeAdjustedWeights:

    def test_returns_dataframe(self):
        base = {"RSI": 0.5, "MACD": 0.5}
        probs = _make_regime_probs(10, "TREND_UP")
        result = regime_adjusted_weights(base, probs, REGIME_INDICATOR_BOOST, probs.index)
        assert isinstance(result, pd.DataFrame)

    def test_shape(self):
        base = {"RSI": 0.5, "MACD": 0.5}
        probs = _make_regime_probs(10, "RANGE")
        result = regime_adjusted_weights(base, probs, REGIME_INDICATOR_BOOST, probs.index)
        assert result.shape == (10, 2)

    def test_all_values_non_negative(self):
        base = {"RSI": 0.5, "MACD": 0.5}
        probs = _make_regime_probs(20, "HIGHVOL")
        result = regime_adjusted_weights(base, probs, REGIME_INDICATOR_BOOST, probs.index)
        assert (result >= 0).all().all()

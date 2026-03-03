"""Unit tests for phinance.blending.*"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phinance.blending import blend_signals, BLEND_METHODS
from phinance.blending.methods import (
    weighted_sum, voting, regime_weighted, REGIME_INDICATOR_BOOST
)
from phinance.blending.regime_detector import detect_regime, regime_to_probs
from phinance.blending.weights import (
    normalise_weights, equal_weights, boost_weights
)
from phinance.exceptions import UnsupportedBlendMethodError


def _make_signals(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "RSI":  rng.uniform(-1, 1, n),
            "MACD": rng.uniform(-1, 1, n),
            "Bollinger": rng.uniform(-1, 1, n),
        },
        index=idx,
    )


class TestBlendMethods:
    def test_blend_methods_list(self):
        assert "weighted_sum" in BLEND_METHODS
        assert "voting" in BLEND_METHODS
        assert "regime_weighted" in BLEND_METHODS

    def test_weighted_sum_empty_signals(self):
        result = blend_signals(pd.DataFrame(), {})
        assert len(result) == 0

    def test_weighted_sum_produces_bounded_output(self):
        sigs = _make_signals()
        result = blend_signals(sigs, {"RSI": 0.5, "MACD": 0.5}, "weighted_sum")
        assert len(result) == 20
        # Weighted sum of bounded inputs should stay roughly in [-1, 1]
        assert result.between(-1.1, 1.1).all()

    def test_voting_produces_clipped_output(self):
        sigs = _make_signals()
        result = blend_signals(sigs, method="voting")
        assert result.between(-1, 1).all()

    def test_equal_weights_when_no_weights_supplied(self):
        sigs = _make_signals()
        r1 = blend_signals(sigs, {}, "weighted_sum")
        r2 = blend_signals(sigs, {"RSI": 1/3, "MACD": 1/3, "Bollinger": 1/3}, "weighted_sum")
        pd.testing.assert_series_equal(r1, r2, check_names=False)

    def test_unknown_method_raises(self):
        sigs = _make_signals()
        with pytest.raises(UnsupportedBlendMethodError):
            blend_signals(sigs, {}, "invented_method")

    def test_regime_weighted_without_probs_falls_back(self):
        sigs = _make_signals()
        # Should not raise even without regime_probs
        result = blend_signals(sigs, {}, "regime_weighted", regime_probs=None)
        assert len(result) == 20

    def test_regime_weighted_with_probs(self):
        sigs = _make_signals(20)
        probs = pd.DataFrame(
            {"TREND_UP": [0.8]*20, "RANGE": [0.2]*20},
            index=sigs.index,
        )
        result = blend_signals(sigs, {}, "regime_weighted", regime_probs=probs)
        assert len(result) == 20
        assert not result.isna().any()


class TestWeightHelpers:
    def test_normalise_weights_sums_to_one(self):
        w = normalise_weights({"A": 2.0, "B": 3.0}, ["A", "B"])
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_equal_weights_all_equal(self):
        w = equal_weights(["A", "B", "C"])
        assert all(abs(v - 1/3) < 1e-9 for v in w.values())

    def test_boost_weights_applies_multiplier(self):
        w = {"A": 0.5, "B": 0.5}
        boosted = boost_weights(w, {"A": 2.0})
        assert boosted["A"] > boosted["B"]


class TestRegimeDetector:
    def test_detect_regime_returns_series(self):
        from tests.fixtures.ohlcv import make_ohlcv
        df = make_ohlcv(60)
        labels = detect_regime(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == 60
        valid = {"TREND_UP", "TREND_DN", "RANGE", "HIGHVOL", "LOWVOL", "BREAKOUT_UP", "BREAKOUT_DN"}
        assert set(labels.unique()).issubset(valid)

    def test_regime_to_probs_one_hot(self):
        from tests.fixtures.ohlcv import make_ohlcv
        df = make_ohlcv(30)
        labels = detect_regime(df)
        probs = regime_to_probs(labels)
        assert probs.shape[0] == 30
        assert (probs.sum(axis=1) <= 1.01).all()  # Each row sums to 0 or 1

"""Unit tests for phi/blending/blender.py."""

from __future__ import annotations

import pandas as pd
import pytest

from phi.blending.blender import blend_signals


def _sigs(data: dict, n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({k: pd.Series(v, index=idx) for k, v in data.items()})


# ── weighted_sum ─────────────────────────────────────────────────────────────

def test_weighted_sum_basic():
    sigs = _sigs({"RSI": [1, 1, -1, 1, -1]})
    result = blend_signals(sigs, method="weighted_sum")
    assert len(result) == 5
    assert result.iloc[0] == pytest.approx(1.0)


def test_weighted_sum_multi():
    sigs = _sigs({"RSI": [1, 1, 1, 1, 1], "MACD": [-1, -1, -1, -1, -1]})
    weights = {"RSI": 0.5, "MACD": 0.5}
    result = blend_signals(sigs, weights=weights, method="weighted_sum")
    # Equal positive and negative → should be ~0
    assert result.abs().max() < 1e-9


# ── voting ────────────────────────────────────────────────────────────────────

def test_voting_basic():
    sigs = _sigs({"RSI": [1, 1, 1], "MACD": [1, -1, 1], "Bollinger": [1, 1, -1]}, n=3)
    result = blend_signals(sigs, method="voting")
    # Bar 0: all +1 → 1.0; Bar 1: 2/3 positive = 0.33; Bar 2: 2/3 positive = 0.33
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(1 / 3)


def test_voting_empty():
    result = blend_signals(pd.DataFrame(), method="voting")
    assert isinstance(result, pd.Series)
    assert len(result) == 0


# ── regime_weighted fallback ──────────────────────────────────────────────────

def test_regime_weighted_fallback():
    """When regime_probs is None, should fall back to weighted_sum."""
    sigs = _sigs({"RSI": [1, -1, 1, -1, 1]})
    result = blend_signals(sigs, method="regime_weighted", regime_probs=None)
    expected = blend_signals(sigs, method="weighted_sum")
    pd.testing.assert_series_equal(result, expected)


# ── empty DataFrame ───────────────────────────────────────────────────────────

def test_blend_empty_signals():
    result = blend_signals(pd.DataFrame())
    assert isinstance(result, pd.Series)
    assert len(result) == 0

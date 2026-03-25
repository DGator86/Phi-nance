"""Regime evaluation metrics (no network)."""

from __future__ import annotations

import pandas as pd

from phi.regime.evaluation import (
    mean_run_length_bars,
    regime_entropy_normalized,
    summarize_regime_metrics,
    transition_rate,
)


def test_regime_entropy_extremes() -> None:
    assert regime_entropy_normalized(pd.Series(["a", "a", "a"])) == 0.0
    s = pd.Series(["a", "b", "c"])
    assert abs(regime_entropy_normalized(s) - 1.0) < 1e-6


def test_transition_rate() -> None:
    s = pd.Series(["a", "a", "b", "b", "a"])
    assert transition_rate(s) > 0


def test_mean_run_length() -> None:
    s = pd.Series(["x", "x", "y", "y", "y"])
    assert abs(mean_run_length_bars(s) - 2.5) < 1e-6


def test_summarize_regime_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.Series(range(100, 130), index=idx, dtype=float)
    ohlcv = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0},
        index=idx,
    )
    lab = pd.Series(["s0"] * 15 + ["s1"] * 15, index=idx)
    m = summarize_regime_metrics(ohlcv, lab)
    assert "regime_entropy_norm" in m
    assert "regime_transition_rate" in m
    assert m["regime_transition_rate"] > 0

"""Unit tests for phi.blending.blender."""

from __future__ import annotations

import pandas as pd
import pytest

from phi.blending import blend_signals


def _signals() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "RSI": [1.0, 0.0, -1.0],
            "MACD": [0.5, 0.5, -0.5],
        },
        index=idx,
    )


def test_weighted_sum_valid() -> None:
    signals = _signals()
    result = blend_signals(signals, method="weighted_sum", weights={"RSI": 0.6, "MACD": 0.4})
    expected = pd.Series([0.8, 0.2, -0.8], index=signals.index, name="composite_signal")
    pd.testing.assert_series_equal(result, expected)


def test_weighted_sum_missing_weights_raises() -> None:
    with pytest.raises(ValueError, match="weights are required"):
        blend_signals(_signals(), method="weighted_sum", weights=None)


def test_weighted_sum_normalizes_weights_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    signals = _signals()
    with caplog.at_level("WARNING", logger="phi.blending.blender"):
        result = blend_signals(signals, method="weighted_sum", weights={"RSI": 2.0, "MACD": 1.0})

    expected = pd.Series([0.8333333333, 0.1666666667, -0.8333333333], index=signals.index, name="composite_signal")
    pd.testing.assert_series_equal(result, expected, rtol=1e-7, atol=1e-7)
    assert "normalizing" in caplog.text


def test_voting_and_out_of_range_warning(caplog: pytest.LogCaptureFixture) -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    signals = pd.DataFrame({"A": [2, 0, -2], "B": [1, -1, 1]}, index=idx)

    with caplog.at_level("WARNING", logger="phi.blending.blender"):
        result = blend_signals(signals, method="voting")

    expected = pd.Series([1, -1, -1], index=idx, name="composite_signal")
    pd.testing.assert_series_equal(result, expected)
    assert "outside [-1, 1]" in caplog.text


def test_regime_weighted_with_explicit_boosts() -> None:
    signals = _signals()
    result = blend_signals(
        signals,
        method="regime_weighted",
        weights={"RSI": 0.5, "MACD": 0.5},
        regime="bull",
        regime_boosts={"bull": {"RSI": 2.0}},
    )
    expected = pd.Series([0.8333333333, 0.1666666667, -0.8333333333], index=signals.index, name="composite_signal")
    pd.testing.assert_series_equal(result, expected, rtol=1e-7, atol=1e-7)


def test_regime_weighted_without_boosts_fallback_logs(caplog: pytest.LogCaptureFixture) -> None:
    signals = _signals()
    with caplog.at_level("WARNING", logger="phi.blending.blender"):
        result = blend_signals(
            signals,
            method="regime_weighted",
            weights={"RSI": 0.5, "MACD": 0.5},
            regime="RANGE",
            regime_boosts=None,
        )

    assert isinstance(result, pd.Series)
    assert "deprecated behavior" in caplog.text


def test_regime_weighted_without_regime_raises() -> None:
    with pytest.raises(ValueError, match="regime is required"):
        blend_signals(
            _signals(),
            method="regime_weighted",
            weights={"RSI": 0.5, "MACD": 0.5},
            regime=None,
            regime_boosts={"bull": {"RSI": 1.2}},
        )


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown blend method"):
        blend_signals(_signals(), method="bad_method", weights={"RSI": 1.0, "MACD": 0.0})


def test_empty_signals_raises() -> None:
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        blend_signals(pd.DataFrame(), method="voting")


def test_signals_all_nan_column_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    signals = pd.DataFrame({"RSI": [None, None, None], "MACD": [1.0, 0.0, -1.0]}, index=idx)
    with pytest.raises(ValueError, match="all-NaN columns"):
        blend_signals(signals, method="weighted_sum", weights={"RSI": 0.5, "MACD": 0.5})


def test_returns_series_with_same_index() -> None:
    signals = _signals()
    result = blend_signals(signals, method="voting")
    assert isinstance(result, pd.Series)
    pd.testing.assert_index_equal(result.index, signals.index)

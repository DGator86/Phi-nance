from __future__ import annotations

import pandas as pd
import pytest

from phi.backtest import build_regime_boosts_from_payload, run_direct_backtest
from phi.exceptions import BacktestError


def _sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.Series(range(100, 130), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )


def test_run_direct_backtest_regime_boosts_accepts_detector_labels() -> None:
    data = _sample_ohlcv()
    indicators = {"Buy & Hold": {"enabled": True, "params": {}}}
    weights = {"Buy & Hold": 1.0}
    regimes = pd.Series(["state_0"] * len(data), index=data.index)

    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators=indicators,
        blend_weights=weights,
        blend_method="regime_weighted",
        regime_series=regimes,
        regime_label_map={"state_0": "bull"},
        regime_boosts={"bull": {"Buy & Hold": 1.2}},
    )

    assert "total_return" in results
    assert len(results["portfolio_value"]) > 1


def test_run_direct_backtest_regime_uses_detector_on_the_fly() -> None:
    class FakeDetector:
        def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
            return pd.Series(["state_0"] * len(ohlcv), index=ohlcv.index)

    data = _sample_ohlcv()
    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators={"Buy & Hold": {"enabled": True, "params": {}}},
        blend_weights={"Buy & Hold": 1.0},
        blend_method="regime_weighted",
        regime_detector=FakeDetector(),
        regime_boosts={"state_0": {"Buy & Hold": 1.0}},
    )
    assert len(results["portfolio_value"]) == len(data) + 1


def test_run_direct_backtest_regime_requires_series_or_detector() -> None:
    with pytest.raises(BacktestError):
        run_direct_backtest(
            ohlcv=_sample_ohlcv(),
            symbol="SPY",
            indicators={"Buy & Hold": {"enabled": True, "params": {}}},
            blend_weights={"Buy & Hold": 1.0},
            blend_method="regime_weighted",
        )


def test_build_regime_boosts_from_payload_maps_friendly_labels() -> None:
    payload = {
        "regime_label_map": {"state_0": "Bull", "state_1": "Bear"},
        "regime_boost_matrix": {
            "state_0": {"RSI": 1.2, "MACD": 0.9},
            "state_1": {"RSI": 0.8, "MACD": 1.1},
        },
    }

    boosts = build_regime_boosts_from_payload(payload)

    assert boosts["Bull"]["RSI"] == 1.2
    assert boosts["Bear"]["MACD"] == 1.1

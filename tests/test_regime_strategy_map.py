from __future__ import annotations

import numpy as np
import pandas as pd

from phi.regime import get_detailed_regime
from phi.regime.regime_definitions import (
    compose_detailed_regime,
    detailed_labels_for_probabilities,
    infer_volatility_regime,
    normalise_regime_probabilities,
)
from phi.regime.strategy_mapping import (
    is_approved_strategy,
    map_regime_probabilities_to_strategies,
    strategies_for_regime,
)


def _synthetic_ohlcv(rows: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(17)
    rets = np.concatenate(
        [
            rng.normal(0.0006, 0.004, rows // 3),
            rng.normal(-0.0004, 0.015, rows // 3),
            rng.normal(0.0002, 0.007, rows - 2 * (rows // 3)),
        ]
    )
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0.002, 0.001, rows)))
    low = close * (1 - np.abs(rng.normal(0.002, 0.001, rows)))
    open_ = close * (1 + rng.normal(0, 0.001, rows))
    volume = rng.integers(800_000, 2_200_000, rows)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_compose_detailed_regime_normalises_labels() -> None:
    detailed = compose_detailed_regime("trend_up", "high_vol")
    assert detailed.label == "BULL_HIGH_VOL"


def test_get_detailed_regime_returns_composite_label() -> None:
    label = get_detailed_regime(_synthetic_ohlcv())
    assert label.startswith(("BULL_", "BEAR_", "RANGING_"))


def test_volatility_regime_is_in_supported_set() -> None:
    vol_regime = infer_volatility_regime(_synthetic_ohlcv(), window=20)
    assert vol_regime in {"LOW_VOL", "NORMAL_VOL", "HIGH_VOL"}


def test_strategy_map_returns_curated_strategies() -> None:
    picked = strategies_for_regime("RANGING_LOW_VOL")
    assert "Iron Condor" in picked
    assert all(is_approved_strategy(name) for name in picked)


def test_probability_mapping_scores_strategies() -> None:
    probs = {
        "BULL_LOW_VOL": 0.7,
        "RANGING_LOW_VOL": 0.3,
    }
    ranked = map_regime_probabilities_to_strategies(probs)
    assert ranked
    assert ranked[0][1] >= ranked[-1][1]


def test_detailed_labels_from_base_probabilities() -> None:
    base = normalise_regime_probabilities({"TREND_UP": 0.8, "RANGE": 0.2})
    detailed = detailed_labels_for_probabilities(base, vol_regime="LOW_VOL")
    assert abs(sum(detailed.values()) - 1.0) < 1e-8
    assert "BULL_LOW_VOL" in detailed

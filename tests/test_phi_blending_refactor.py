from __future__ import annotations

import pandas as pd

from phi.blending import BLEND_METHODS, blend_signals


def _signals() -> pd.DataFrame:
    return pd.DataFrame({"RSI": [0.5, -0.3, 0.8], "MACD": [0.2, 0.1, -0.4]})


def test_weighted_sum_matches_expected() -> None:
    signals = _signals()
    result = blend_signals(signals, method="weighted_sum", weights={"RSI": 0.6, "MACD": 0.4})
    expected = 0.6 * signals["RSI"] + 0.4 * signals["MACD"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_ai_driven_registered_and_runs() -> None:
    assert "ai_driven" in BLEND_METHODS
    result = blend_signals(_signals(), method="ai_driven", auto_tune=True)
    assert len(result) == 3
    assert not result.isna().any()

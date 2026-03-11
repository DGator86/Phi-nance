from __future__ import annotations

import pandas as pd

from phi.options.data_adapter import adapt_for_backtesting


def test_adapter_maps_and_adds_expected_columns():
    raw = pd.DataFrame(
        {
            "quote_time": [1_672_531_200_000],
            "last": [2.5],
            "total_volume": [123],
            "strike_price": [400],
            "expiration_date": ["2023-02-17"],
        }
    )

    out = adapt_for_backtesting(raw)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert "close" in out.columns
    assert "volume" in out.columns
    assert "strike" in out.columns
    assert "expiration" in out.columns
    assert "delta" in out.columns
    assert "open_interest" in out.columns


def test_adapter_handles_missing_quote_time_and_missing_fields():
    raw = pd.DataFrame({"last": [1.0], "total_volume": [10]}, index=["2023-01-01T00:00:00"])

    out = adapt_for_backtesting(raw)

    assert isinstance(out.index, pd.DatetimeIndex)
    assert out["close"].iloc[0] == 1.0
    assert out["volume"].iloc[0] == 10
    assert "gamma" in out.columns
    assert "expiration" in out.columns

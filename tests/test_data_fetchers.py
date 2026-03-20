from __future__ import annotations

import pandas as pd
import pytest

from phi.data import fetchers


def test_fetch_falls_back_to_yfinance_for_unknown_vendor(monkeypatch):
    expected = pd.DataFrame({"close": [1.0]})
    monkeypatch.setattr(fetchers, "fetch_yfinance", lambda *args, **kwargs: expected)

    out = fetchers.fetch("SPY", "2024-01-01", "2024-01-02", vendor="unknown")

    pd.testing.assert_frame_equal(out, expected)


def test_dataset_summary_returns_expected_fields():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {"open": [100, 101], "high": [101, 102], "low": [99, 100], "close": [100, 110], "volume": [10, 20]},
        index=idx,
    )

    summary = fetchers.dataset_summary(df, "spy")

    assert summary["symbol"] == "SPY"
    assert summary["rows"] == 2
    assert summary["total_return"] == pytest.approx(0.1)

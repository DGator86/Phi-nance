from __future__ import annotations

import pandas as pd

from app_streamlit import cache as app_cache


def test_load_historical_data_calls_fetch_and_cache(monkeypatch):
    expected = pd.DataFrame({"close": [1.0, 2.0]})

    monkeypatch.setattr(app_cache, "fetch_and_cache", lambda *args, **kwargs: expected)

    out = app_cache.load_historical_data.__wrapped__("SPY", "2024-01-01", "2024-01-03", "1D", "yfinance")

    pd.testing.assert_frame_equal(out, expected)


def test_compute_indicator_signals_calls_compute_indicator(monkeypatch):
    data = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    expected = pd.Series([1, 0, -1])

    monkeypatch.setattr(app_cache, "compute_indicator", lambda *args, **kwargs: expected)

    out = app_cache.compute_indicator_signals.__wrapped__(data, "RSI", {"period": 14})

    pd.testing.assert_series_equal(out, expected)

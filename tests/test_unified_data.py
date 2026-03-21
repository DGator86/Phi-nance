"""Tests for phi.data.unified_data."""

from __future__ import annotations

import pandas as pd
from unittest.mock import patch

from phi.data import unified_data as ud


def test_get_ohlcv_delegates_to_fetch_and_cache() -> None:
    want = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]})
    with patch("phi.data.unified_data.fetch_and_cache", return_value=want) as m:
        out = ud.get_ohlcv("spy", "2020-01-01", "2020-02-01", "1D", "yfinance")
        assert out is want
        m.assert_called_once_with(
            "yfinance",
            "spy",
            "1D",
            "2020-01-01",
            "2020-02-01",
            force_refresh=False,
            fallback_vendors=None,
        )


def test_load_ohlcv_cached_first_hits_cache() -> None:
    cached = pd.DataFrame({"open": [1.0]})
    with patch("phi.data.unified_data.get_cached_dataset", return_value=cached):
        with patch("phi.data.unified_data.fetch_and_cache") as fetch:
            out = ud.load_ohlcv_cached_first("QQQ", "2021-01-01", "2021-06-01", "1D", "yfinance")
            assert out is cached
            fetch.assert_not_called()


def test_try_external_ohlcv_missing_spec_returns_none() -> None:
    assert ud.try_external_ohlcv("ETH", "2024-01-01", "2024-02-01", "1H") is None


def test_get_ohlcv_with_optional_hook_primary_ok() -> None:
    df = pd.DataFrame({"a": [1]})
    with patch("phi.data.unified_data.get_ohlcv", return_value=df):
        out = ud.get_ohlcv_with_optional_hook("X", "2020-01-01", "2020-02-01")
        assert out is df

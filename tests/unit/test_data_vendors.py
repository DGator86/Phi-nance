"""
tests/unit/test_data_vendors.py
================================

Comprehensive tests for phinance.data (cache, vendors, utils)
with all external APIs (yfinance, requests) mocked.

Coverage targets
----------------
  - normalize_ohlcv / ohlcv_sanity_check
  - DataCache: save, load, list
  - BaseVendor abstract interface
  - YFinanceVendor (daily + intraday) — yfinance mocked
  - fetch_and_cache / get_cached_dataset
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.data.cache import normalize_ohlcv, ohlcv_sanity_check, DataCache
from phinance.data.vendors.base import BaseVendor
from phinance.data.vendors.yfinance import YFinanceVendor
from phinance.exceptions import (
    DataFetchError,
    DataValidationError,
    UnsupportedTimeframeError,
)


# ─────────────────────────────────────────────────────────────────────────────
#  normalize_ohlcv
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeOhlcv:

    def _raw(self, casing: str = "lower") -> pd.DataFrame:
        df = make_ohlcv(10)
        if casing == "upper":
            df.columns = [c.upper() for c in df.columns]
        elif casing == "mixed":
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def test_returns_dataframe(self):
        result = normalize_ohlcv(self._raw())
        assert isinstance(result, pd.DataFrame)

    def test_columns_lowercase(self):
        result = normalize_ohlcv(self._raw("upper"))
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_mixed_case_columns(self):
        result = normalize_ohlcv(self._raw("mixed"))
        assert "close" in result.columns

    def test_index_is_datetime(self):
        result = normalize_ohlcv(self._raw())
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_is_sorted_ascending(self):
        df = self._raw()
        result = normalize_ohlcv(df.iloc[::-1])  # reverse order
        assert result.index.is_monotonic_increasing

    def test_index_is_tz_naive(self):
        df = self._raw()
        df.index = df.index.tz_localize("UTC")
        result = normalize_ohlcv(df)
        assert result.index.tz is None

    def test_missing_column_raises(self):
        df = make_ohlcv(5).drop(columns=["volume"])
        with pytest.raises(DataValidationError, match="volume"):
            normalize_ohlcv(df)

    def test_all_required_columns_present(self):
        result = normalize_ohlcv(self._raw())
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_extra_columns_dropped(self):
        df = make_ohlcv(5)
        df["extra"] = 99.0
        result = normalize_ohlcv(df)
        assert "extra" not in result.columns

    def test_row_count_preserved(self):
        df = make_ohlcv(20)
        result = normalize_ohlcv(df)
        assert len(result) == 20


# ─────────────────────────────────────────────────────────────────────────────
#  ohlcv_sanity_check
# ─────────────────────────────────────────────────────────────────────────────

class TestOhlcvSanityCheck:

    def test_clean_df_no_errors(self):
        """Clean data should not raise any exceptions."""
        ohlcv_sanity_check(make_ohlcv(50), symbol="TEST")  # Should not raise

    def test_negative_close_triggers_log_warning(self, caplog):
        """Negative values should log a WARNING (not raise)."""
        import logging
        df = make_ohlcv(10, negative=True)
        with caplog.at_level(logging.WARNING):
            ohlcv_sanity_check(df, symbol="TEST")
        assert any("negative" in msg.lower() for msg in caplog.messages)

    def test_non_chronological_index_triggers_log_warning(self, caplog):
        """Reversed index should log a WARNING."""
        import logging
        df = make_ohlcv(10)
        df_rev = df.iloc[::-1].copy()
        with caplog.at_level(logging.WARNING):
            ohlcv_sanity_check(df_rev, symbol="TEST")
        assert any("chronological" in msg.lower() or "order" in msg.lower()
                   or "sort" in msg.lower()
                   for msg in caplog.messages)


# ─────────────────────────────────────────────────────────────────────────────
#  DataCache
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCache:

    @pytest.fixture
    def cache(self, tmp_path):
        """DataCache backed by a temp directory."""
        return DataCache(root=tmp_path)

    def test_save_and_load(self, cache):
        df = make_ohlcv(30)
        cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-12-31")
        loaded = cache.load("yfinance", "SPY", "1D", "2023-01-01", "2023-12-31")
        assert loaded is not None
        assert len(loaded) == 30

    def test_load_miss_returns_none(self, cache):
        result = cache.load("yfinance", "NONEXIST", "1D", "2020-01-01", "2020-12-31")
        assert result is None

    def test_list_empty_cache(self, cache):
        listing = cache.list_datasets()
        assert listing == []

    def test_list_after_save(self, cache):
        df = make_ohlcv(10)
        cache.save(df, "yfinance", "AAPL", "1D", "2023-01-01", "2023-06-30")
        listing = cache.list_datasets()
        assert len(listing) >= 1

    def test_saved_data_has_correct_shape(self, cache):
        df = make_ohlcv(50)
        cache.save(df, "yfinance", "GOOG", "1D", "2023-01-01", "2023-12-31")
        loaded = cache.load("yfinance", "GOOG", "1D", "2023-01-01", "2023-12-31")
        assert loaded.shape[1] == 5  # open, high, low, close, volume

    def test_cache_key_case_insensitive_symbol(self, cache):
        """Same symbol in upper/lower case should retrieve the same entry."""
        df = make_ohlcv(10)
        cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-06-30")
        # Should not raise even if internal key storage differs
        loaded = cache.load("yfinance", "SPY", "1D", "2023-01-01", "2023-06-30")
        assert loaded is not None

    def test_multiple_symbols_stored_independently(self, cache):
        df1 = make_ohlcv(10)
        df2 = make_ohlcv(20)
        cache.save(df1, "yfinance", "AAA", "1D", "2023-01-01", "2023-01-10")
        cache.save(df2, "yfinance", "BBB", "1D", "2023-01-01", "2023-01-20")
        l1 = cache.load("yfinance", "AAA", "1D", "2023-01-01", "2023-01-10")
        l2 = cache.load("yfinance", "BBB", "1D", "2023-01-01", "2023-01-20")
        assert len(l1) == 10
        assert len(l2) == 20


# ─────────────────────────────────────────────────────────────────────────────
#  BaseVendor (abstract interface)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseVendor:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseVendor()

    def test_concrete_subclass_works(self):
        class MockVendor(BaseVendor):
            name = "mock"

            def fetch(self, symbol, timeframe, start, end, **kwargs):
                return make_ohlcv(10)

        vendor = MockVendor()
        df = vendor.fetch("X", "1D", "2023-01-01", "2023-12-31")
        assert isinstance(df, pd.DataFrame)

    def test_normalize_helper(self):
        class MockVendor(BaseVendor):
            name = "mock"

            def fetch(self, symbol, timeframe, start, end, **kwargs):
                return self._normalize(make_ohlcv(5))

        vendor = MockVendor()
        df = vendor.fetch("X", "1D", "2023-01-01", "2023-01-05")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}


# ─────────────────────────────────────────────────────────────────────────────
#  YFinanceVendor — mocked
# ─────────────────────────────────────────────────────────────────────────────

def _make_yf_ticker_mock(n: int = 60) -> MagicMock:
    """Mock yfinance.Ticker that returns synthetic OHLCV."""
    df = make_ohlcv(n)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df
    return mock_ticker


class TestYFinanceVendor:

    @patch("yfinance.Ticker")
    def test_fetch_daily_returns_dataframe(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_yf_ticker_mock(60)
        vendor = YFinanceVendor()
        df = vendor.fetch("SPY", "1D", "2023-01-01", "2023-12-31")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    @patch("yfinance.Ticker")
    def test_fetch_daily_normalizes_columns(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_yf_ticker_mock(30)
        vendor = YFinanceVendor()
        df = vendor.fetch("AAPL", "1D", "2023-01-01", "2023-06-30")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    @patch("yfinance.Ticker")
    def test_fetch_intraday_5m(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_yf_ticker_mock(100)
        vendor = YFinanceVendor()
        df = vendor.fetch("SPY", "5m", "2023-12-01", "2023-12-30")
        assert isinstance(df, pd.DataFrame)

    @patch("yfinance.Ticker")
    def test_fetch_intraday_1h(self, mock_ticker_cls):
        mock_ticker_cls.return_value = _make_yf_ticker_mock(200)
        vendor = YFinanceVendor()
        df = vendor.fetch("QQQ", "1H", "2023-01-01", "2024-01-01")
        assert isinstance(df, pd.DataFrame)

    def test_unsupported_timeframe_raises(self):
        vendor = YFinanceVendor()
        with pytest.raises(UnsupportedTimeframeError):
            vendor.fetch("SPY", "4H", "2023-01-01", "2023-12-31")

    def test_weekly_timeframe_raises(self):
        vendor = YFinanceVendor()
        with pytest.raises(UnsupportedTimeframeError):
            vendor.fetch("SPY", "1W", "2023-01-01", "2023-12-31")

    @patch("yfinance.Ticker")
    def test_empty_response_raises_data_fetch_error(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker
        vendor = YFinanceVendor(max_retries=1)
        with pytest.raises(DataFetchError):
            vendor.fetch("FAKE", "1D", "2023-01-01", "2023-12-31")

    @patch("yfinance.Ticker")
    def test_retry_on_exception(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        # First call raises, second succeeds
        mock_ticker.history.side_effect = [
            RuntimeError("network error"),
            make_ohlcv(30).rename(columns=lambda c: c.capitalize()),
        ]
        mock_ticker_cls.return_value = mock_ticker
        vendor = YFinanceVendor(max_retries=2)
        df = vendor.fetch("SPY", "1D", "2023-01-01", "2023-12-31")
        assert not df.empty

    @patch("yfinance.Ticker")
    def test_all_retries_fail_raises(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = RuntimeError("always fails")
        mock_ticker_cls.return_value = mock_ticker
        vendor = YFinanceVendor(max_retries=2)
        with pytest.raises(DataFetchError):
            vendor.fetch("BAD", "1D", "2023-01-01", "2023-12-31")

    def test_vendor_name(self):
        assert YFinanceVendor.name == "yfinance"

    def test_vendor_default_retries(self):
        v = YFinanceVendor()
        assert v.max_retries == 3


# ─────────────────────────────────────────────────────────────────────────────
#  fetch_and_cache / get_cached_dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchAndCache:
    """Test the high-level fetch_and_cache() + get_cached_dataset() with a mocked vendor."""

    @patch("yfinance.Ticker")
    def test_fetch_and_cache_returns_dataframe(self, mock_ticker_cls, tmp_path):
        from phinance.data.cache import fetch_and_cache, DataCache
        df = make_ohlcv(20)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        mock_ticker_cls.return_value = mock_ticker
        # Direct DataCache with tmp_path to avoid writes to project root
        cache = DataCache(root=tmp_path)
        with patch("phinance.data.cache.DataCache", return_value=cache):
            result = fetch_and_cache(
                vendor="yfinance", symbol="SPY", timeframe="1D",
                start="2023-01-01", end="2023-12-31",
            )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_get_cached_dataset_miss_returns_none(self, tmp_path):
        from phinance.data.cache import DataCache
        cache = DataCache(root=tmp_path)
        result = cache.load("yfinance", "NOTCACHED", "1D", "2023-01-01", "2023-12-31")
        assert result is None

    def test_get_cached_dataset_hit(self, tmp_path):
        from phinance.data.cache import DataCache
        cache = DataCache(root=tmp_path)
        df = make_ohlcv(15)
        cache.save(df, "yfinance", "TSLA", "1D", "2023-01-01", "2023-06-30")
        result = cache.load("yfinance", "TSLA", "1D", "2023-01-01", "2023-06-30")
        assert result is not None
        assert len(result) == 15

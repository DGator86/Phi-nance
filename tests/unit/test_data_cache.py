"""Unit tests for phinance.data.cache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv
from phinance.data.cache import (
    DataCache,
    normalize_ohlcv,
    ohlcv_sanity_check,
    fetch_and_cache,
)
from phinance.exceptions import DataValidationError


# ── normalize_ohlcv ───────────────────────────────────────────────────────────

class TestNormalizeOhlcv:
    def test_valid_dataframe_returns_correct_columns(self):
        df = make_ohlcv(5)
        result = normalize_ohlcv(df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        assert len(result) == 5

    def test_uppercase_columns_are_normalized(self):
        df = make_ohlcv(5)
        df.columns = [c.upper() for c in df.columns]
        result = normalize_ohlcv(df)
        assert "close" in result.columns

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(DataValidationError, match="missing required columns"):
            normalize_ohlcv(df)

    def test_index_is_sorted_ascending(self):
        df = make_ohlcv(10)
        df = df.iloc[::-1]  # reverse
        result = normalize_ohlcv(df)
        assert result.index.is_monotonic_increasing


# ── ohlcv_sanity_check ────────────────────────────────────────────────────────

class TestOhlcvSanityCheck:
    def test_negative_prices_trigger_warning(self, caplog):
        import logging
        df = make_ohlcv(5, negative=True)
        with caplog.at_level(logging.WARNING, logger="phinance"):
            ohlcv_sanity_check(df, symbol="TEST")
        assert any("negative" in m.lower() for m in caplog.messages)

    def test_non_chronological_index_triggers_warning(self, caplog):
        import logging
        df = make_ohlcv(5)
        df = df.iloc[::-1]  # reverse
        with caplog.at_level(logging.WARNING, logger="phinance"):
            ohlcv_sanity_check(df, symbol="TEST")
        assert any("chronological" in m.lower() for m in caplog.messages)

    def test_clean_df_triggers_no_warnings(self, caplog):
        import logging
        df = make_ohlcv(5)
        with caplog.at_level(logging.WARNING, logger="phinance"):
            ohlcv_sanity_check(df, symbol="SPY")
        assert len(caplog.messages) == 0


# ── DataCache ─────────────────────────────────────────────────────────────────

class TestDataCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        cache = DataCache(root=tmp_path)
        df = make_ohlcv(10)
        cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
        loaded = cache.load("yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
        assert loaded is not None
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)

    def test_exists_returns_false_before_save(self, tmp_path):
        cache = DataCache(root=tmp_path)
        assert not cache.exists("yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")

    def test_exists_returns_true_after_save(self, tmp_path):
        cache = DataCache(root=tmp_path)
        df = make_ohlcv(5)
        cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-05")
        assert cache.exists("yfinance", "SPY", "1D", "2023-01-01", "2023-01-05")

    def test_list_datasets_returns_saved_entry(self, tmp_path):
        cache = DataCache(root=tmp_path)
        df = make_ohlcv(5)
        cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-05")
        datasets = cache.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["symbol"] == "SPY"


# ── fetch_and_cache cache-hit ─────────────────────────────────────────────────

class TestFetchAndCacheCacheHit:
    def test_cache_hit_skips_vendor_fetch(self, monkeypatch, tmp_path):
        cache = DataCache(root=tmp_path)
        df = make_ohlcv(5)
        cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-05")

        mock_vendor = MagicMock(side_effect=AssertionError("Should not call fetch"))
        monkeypatch.setattr("phinance.data.cache._DATA_CACHE_ROOT", tmp_path)

        result = fetch_and_cache("yfinance", "SPY", "1D", "2023-01-01", "2023-01-05")
        assert result is not None
        assert len(result) == 5

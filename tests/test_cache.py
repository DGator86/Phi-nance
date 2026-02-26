"""Unit tests for phi/data/cache.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from phi.data.cache import (
    DataCache,
    _normalize_ohlcv,
    _ohlcv_sanity_check,
    fetch_and_cache,
)


def _make_ohlcv(n: int = 5, negative: bool = False) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame."""
    close = [100.0 + i for i in range(n)]
    if negative:
        close[2] = -1.0
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [1000] * n},
        index=idx,
    )


# ── _normalize_ohlcv ─────────────────────────────────────────────────────────

def test_normalize_ohlcv_valid():
    df = _make_ohlcv()
    result = _normalize_ohlcv(df)
    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == 5


def test_normalize_ohlcv_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        _normalize_ohlcv(df)


# ── DataCache save / load ────────────────────────────────────────────────────

def test_datacache_save_and_load(tmp_path):
    cache = DataCache(root=tmp_path)
    df = _make_ohlcv()
    cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
    loaded = cache.load("yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
    assert loaded is not None
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)


def test_datacache_exists(tmp_path):
    cache = DataCache(root=tmp_path)
    assert not cache.exists("yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
    df = _make_ohlcv()
    cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
    assert cache.exists("yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")


# ── fetch_and_cache uses cache ───────────────────────────────────────────────

def test_fetch_and_cache_uses_cache(monkeypatch, tmp_path):
    """When a cache hit exists, no external fetch should be called."""
    cache = DataCache(root=tmp_path)
    df = _make_ohlcv()
    cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")

    mock_fetch = MagicMock(side_effect=AssertionError("Should not call fetch"))
    monkeypatch.setattr("phi.data.cache._fetch_from_yfinance", mock_fetch)

    # Temporarily redirect DataCache root
    monkeypatch.setattr("phi.data.cache._DATA_CACHE_ROOT", tmp_path)

    result = fetch_and_cache("yfinance", "SPY", "1D", "2023-01-01", "2023-01-10")
    assert result is not None
    mock_fetch.assert_not_called()


# ── OHLCV sanity checks ──────────────────────────────────────────────────────

def test_ohlcv_sanity_no_negative_prices(caplog):
    import logging
    df = _make_ohlcv(negative=True)
    with caplog.at_level(logging.WARNING, logger="phi.data.cache"):
        _ohlcv_sanity_check(df, symbol="TEST")
    assert any("negative" in m.lower() for m in caplog.messages)


def test_ohlcv_sanity_non_chronological(caplog):
    import logging
    df = _make_ohlcv()
    df = df.iloc[::-1]  # Reverse the index → not monotonic increasing
    with caplog.at_level(logging.WARNING, logger="phi.data.cache"):
        _ohlcv_sanity_check(df, symbol="TEST")
    assert any("chronological" in m.lower() for m in caplog.messages)

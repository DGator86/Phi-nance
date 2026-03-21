from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from phi.data import cache as cache_mod
from phi.data.cache import CacheCorruptedError, DataCache, DataFetchError, fetch_and_cache, is_cache_stale


def test_get_fetcher_accepts_unusual_whales_aliases() -> None:
    from phi.data.cache import _get_fetcher

    f1 = _get_fetcher("unusual_whales")
    f2 = _get_fetcher("Unusual Whales")
    f3 = _get_fetcher("unusualwhales")
    assert callable(f1) and f1 is f2 is f3


def _make_ohlcv(rows: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    base = [100.0 + i for i in range(rows)]
    return pd.DataFrame(
        {
            "open": base,
            "high": [v + 1 for v in base],
            "low": [v - 1 for v in base],
            "close": base,
            "volume": [1_000.0] * rows,
        },
        index=idx,
    )


def test_fetch_and_cache_cache_hit_skips_vendor(monkeypatch, tmp_path):
    cache = DataCache(root=tmp_path)
    df = _make_ohlcv()
    cache.save(df, "yfinance", "SPY", "1D", "2024-01-01", "2024-01-05")

    monkeypatch.setattr(cache_mod, "DataCache", lambda: DataCache(root=tmp_path))
    called = {"fetch": 0}

    def fake_fetch(*_args, **_kwargs):
        called["fetch"] += 1
        return _make_ohlcv()

    monkeypatch.setattr(cache_mod, "_fetch_with_retry", fake_fetch)

    out = fetch_and_cache("yfinance", "SPY", "1D", "2024-01-01", "2024-01-05")

    assert called["fetch"] == 0
    pd.testing.assert_frame_equal(out, df, check_freq=False)


def test_fetch_and_cache_stale_cache_refetches(monkeypatch, tmp_path):
    cache = DataCache(root=tmp_path)
    old = _make_ohlcv(3)
    cache.save(old, "yfinance", "SPY", "1D", "2024-01-01", "2024-01-03")

    cache_path = cache._parquet_path("yfinance", "SPY", "1D", "2024-01-01", "2024-01-03")
    meta_path = cache_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["fetch_timestamp"] = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    monkeypatch.setattr(cache_mod, "DataCache", lambda: DataCache(root=tmp_path))
    new_df = _make_ohlcv(4)
    monkeypatch.setattr(cache_mod, "_fetch_with_retry", lambda *_a, **_k: new_df)

    out = fetch_and_cache("yfinance", "SPY", "1D", "2024-01-01", "2024-01-03")
    assert len(out) == 4


def test_cache_load_raises_corrupted_for_bad_parquet(tmp_path):
    cache = DataCache(root=tmp_path)
    path = cache._parquet_path("yfinance", "SPY", "1D", "2024-01-01", "2024-01-05")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-a-parquet", encoding="utf-8")
    path.with_suffix(".meta.json").write_text('{"fetch_timestamp":"2024-01-01T00:00:00Z"}', encoding="utf-8")

    with pytest.raises(CacheCorruptedError):
        cache.load("yfinance", "SPY", "1D", "2024-01-01", "2024-01-05")


def test_is_cache_stale_true_for_invalid_metadata(tmp_path):
    path = tmp_path / "demo.parquet"
    path.write_text("dummy", encoding="utf-8")
    path.with_suffix(".meta.json").write_text("{invalid json", encoding="utf-8")

    assert is_cache_stale(path, "1D") is True


def test_fetch_and_cache_wraps_fetch_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "DataCache", lambda: DataCache(root=tmp_path))

    def boom(*_args, **_kwargs):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(cache_mod, "_fetch_with_retry", boom)

    with pytest.raises(DataFetchError, match="Failed to fetch data"):
        fetch_and_cache("yfinance", "SPY", "1D", "2024-01-01", "2024-01-05", force_refresh=True)


def test_fetch_and_cache_uses_fallback_when_primary_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "DataCache", lambda: DataCache(root=tmp_path))

    def fake_get_fetcher(vendor):
        if vendor == "yfinance":
            return "yf"
        if vendor == "alphavantage":
            return "av"
        raise ValueError(vendor)

    def fake_fetch(fetcher, *_args, **_kwargs):
        if fetcher == "yf":
            raise RuntimeError("yfinance unavailable")
        return _make_ohlcv(2)

    monkeypatch.setattr(cache_mod, "_get_fetcher", fake_get_fetcher)
    monkeypatch.setattr(cache_mod, "_fetch_with_retry", fake_fetch)
    monkeypatch.setattr(cache_mod.time, "sleep", lambda *_args, **_kwargs: None)

    out = fetch_and_cache(
        "yfinance",
        "SPY",
        "1D",
        "2024-01-01",
        "2024-01-05",
        force_refresh=True,
        fallback_vendors=["alphavantage"],
    )

    assert len(out) == 2
    meta = cache_mod.load_metadata("yfinance", "SPY", "1D", "2024-01-01", "2024-01-05")
    assert meta is not None
    assert meta["vendors_used"] == ["yfinance", "alphavantage"]


def test_fetch_and_cache_fills_missing_dates_with_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "DataCache", lambda: DataCache(root=tmp_path))

    primary = _make_ohlcv(5).drop(pd.Timestamp("2024-01-03"))
    fallback = _make_ohlcv(1).set_index(pd.DatetimeIndex([pd.Timestamp("2024-01-03")]))

    def fake_get_fetcher(vendor):
        return vendor

    def fake_fetch(fetcher, _symbol, start_s, _end_s):
        if fetcher == "yfinance":
            return primary
        if fetcher == "alphavantage" and start_s == "2024-01-03":
            return fallback
        return pd.DataFrame()

    monkeypatch.setattr(cache_mod, "_get_fetcher", fake_get_fetcher)
    monkeypatch.setattr(cache_mod, "_fetch_with_retry", fake_fetch)
    monkeypatch.setattr(cache_mod.time, "sleep", lambda *_args, **_kwargs: None)

    out = fetch_and_cache(
        "yfinance",
        "SPY",
        "1D",
        "2024-01-01",
        "2024-01-05",
        force_refresh=True,
        fallback_vendors=["alphavantage"],
    )

    assert pd.Timestamp("2024-01-03") in out.index
    assert len(out) == 5

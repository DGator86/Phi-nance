"""
Phi-nance Data Cache
====================

Stores historical OHLCV in parquet format at:
  /data_cache/{vendor}/{symbol}/{timeframe}/{start}_{end}.parquet

Each dataset has metadata.json alongside it.
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Project root: parent of phi/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_CACHE_ROOT = _PROJECT_ROOT / "data_cache"


def _ensure_cache_dir() -> Path:
    _DATA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    return _DATA_CACHE_ROOT


def _cache_path(vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
    """Path to parquet file for this dataset."""
    s = str(start)[:10].replace("-", "")
    e = str(end)[:10].replace("-", "")
    base = _ensure_cache_dir() / vendor / symbol.upper() / timeframe
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{s}_{e}.parquet"


def _metadata_path(vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
    """Path to metadata.json for this dataset."""
    p = _cache_path(vendor, symbol, timeframe, start, end)
    return p.with_suffix(".parquet.metadata.json")


class DataCache:
    """
    Manages parquet-based dataset cache.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _DATA_CACHE_ROOT

    def get_path(self, vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
        return _cache_path(vendor, symbol, timeframe, start, end)

    def exists(self, vendor: str, symbol: str, timeframe: str, start: str, end: str) -> bool:
        p = self.get_path(vendor, symbol, timeframe, start, end)
        return p.exists()

    def load(
        self,
        vendor: str,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Load cached dataset if it exists."""
        p = self.get_path(vendor, symbol, timeframe, start, end)
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def save(
        self,
        df: pd.DataFrame,
        vendor: str,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save dataset and metadata."""
        p = self.get_path(vendor, symbol, timeframe, start, end)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=True)

        meta = metadata or {}
        meta.setdefault("vendor", vendor)
        meta.setdefault("symbol", symbol)
        meta.setdefault("timeframe", timeframe)
        meta.setdefault("start", str(start)[:10])
        meta.setdefault("end", str(end)[:10])
        meta.setdefault("rows", len(df))
        meta.setdefault("columns", list(df.columns))
        meta_path = _metadata_path(vendor, symbol, timeframe, start, end)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return p

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all cached datasets with metadata."""
        results = []
        if not self.root.exists():
            return results
        for parquet in self.root.rglob("*.parquet"):
            meta_path = Path(str(parquet) + ".metadata.json")
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            rel = parquet.relative_to(self.root)
            parts = rel.parts
            if len(parts) >= 4:
                meta.setdefault("vendor", parts[0])
                meta.setdefault("symbol", parts[1])
                meta.setdefault("timeframe", parts[2])
                meta.setdefault("path", str(parquet))
            results.append(meta)
        return results


def _fetch_yfinance(symbol: str, timeframe: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch OHLCV from yfinance and return normalised DataFrame."""
    import yfinance as yf

    tf_map = {"1D": "1d", "4H": "1h", "1H": "1h", "15m": "15m", "5m": "5m", "1m": "1m"}
    interval = tf_map.get(timeframe, "1d")
    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start_s, end=end_s, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yFinance: no data for {symbol}")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _fetch_massive(symbol: str, timeframe: str, start_s: str, end_s: str, api_key: str) -> pd.DataFrame:
    """Fetch OHLCV from Massive.com (formerly Polygon.io, rebranded Oct 2025).
    Uses the v2 aggregates REST endpoint directly with requests.
    Handles next_url pagination automatically.
    Free tier: unlimited calls, 15-min delayed data.
    """
    import requests

    tf_map = {
        "1D":  ("day",    1),
        "4H":  ("hour",   4),
        "1H":  ("hour",   1),
        "15m": ("minute", 15),
        "5m":  ("minute", 5),
        "1m":  ("minute", 1),
    }
    timespan, multiplier = tf_map.get(timeframe, ("day", 1))

    url = (
        f"https://api.massive.com/v2/aggs/ticker/{symbol}/range"
        f"/{multiplier}/{timespan}/{start_s}/{end_s}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}
    rows = []

    while url:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        status = data.get("status", "")
        if status not in ("OK", "DELAYED"):
            raise ValueError(f"Massive: {data.get('status', 'error')} — {data.get('error', data.get('message', ''))}")
        rows.extend(data.get("results", []))
        url = data.get("next_url")
        params = {"apiKey": api_key}  # next_url already has other params baked in

    if not rows:
        raise ValueError(f"Massive: no data for {symbol}")

    df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "ts"})
    df.index = pd.to_datetime(df["ts"], unit="ms")
    df.index.name = "datetime"
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def _fetch_stockdata(symbol: str, start_s: str, end_s: str, api_key: str) -> pd.DataFrame:
    """Fetch EOD OHLCV from StockData.org (free: ~100 req/day, 1yr history).
    Paginates automatically — 50 records per page.
    Only supports daily timeframe (EOD endpoint).
    """
    import requests

    base_url = "https://api.stockdata.org/v1/data/eod"
    rows = []
    page = 1

    while True:
        resp = requests.get(base_url, params={
            "symbols": symbol,
            "date_from": start_s,
            "date_to": end_s,
            "sort": "asc",
            "page": page,
            "api_token": api_key,
        }, timeout=20)
        data = resp.json()

        if "error" in data:
            raise ValueError(f"StockData.org: {data['error'].get('message', data['error'])}")

        batch = data.get("data", [])
        if not batch:
            break

        rows.extend(batch)

        meta = data.get("meta", {})
        # Stop if we've received everything
        if len(rows) >= meta.get("found", len(rows)):
            break
        page += 1

    if not rows:
        raise ValueError(f"StockData.org: no data for {symbol}")

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_finnhub(symbol: str, timeframe: str, start_s: str, end_s: str, api_key: str) -> pd.DataFrame:
    """Fetch OHLCV from Finnhub REST API. Free tier: 60 req/min."""
    import requests
    import time as _time

    res_map = {"1D": "D", "4H": "60", "1H": "60", "15m": "15", "5m": "5", "1m": "1"}
    resolution = res_map.get(timeframe, "D")

    from_ts = int(pd.Timestamp(start_s).timestamp())
    to_ts = int(pd.Timestamp(end_s).timestamp())

    url = (
        f"https://finnhub.io/api/v1/stock/candle"
        f"?symbol={symbol}&resolution={resolution}"
        f"&from={from_ts}&to={to_ts}&token={api_key}"
    )
    resp = requests.get(url, timeout=20)
    data = resp.json()

    if data.get("s") != "ok":
        raise ValueError(f"Finnhub: {data.get('s', 'error')} for {symbol} — check API key or symbol")

    df = pd.DataFrame({
        "open":   data["o"],
        "high":   data["h"],
        "low":    data["l"],
        "close":  data["c"],
        "volume": data["v"],
    }, index=pd.to_datetime(data["t"], unit="s"))
    df.index.name = "datetime"
    return df


def fetch_and_cache(
    vendor: str,
    symbol: str,
    timeframe: str,
    start: str | date | datetime,
    end: str | date | datetime,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV from vendor and cache. Returns DataFrame.
    Reuses cache if available and fresh.
    Supported vendors: alphavantage, yfinance, finnhub.
    """
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    vendor_l = vendor.lower().replace(" ", "").replace("_", "")
    cache = DataCache()
    existing = cache.load(vendor, symbol.upper(), timeframe, start_s, end_s)
    if existing is not None and len(existing) > 0:
        return existing

    if vendor_l in ("alphavantage", "alphavantagefixed"):
        # For 1D, use yfinance (AV intraday is ~30 days max on free tier)
        if timeframe == "1D":
            df = _fetch_yfinance(symbol, timeframe, start_s, end_s)
        else:
            from regime_engine.data_fetcher import AlphaVantageFetcher
            av = AlphaVantageFetcher(api_key=api_key)
            interval_map = {"4H": "60min", "1H": "60min", "15m": "15min", "5m": "5min", "1m": "1min"}
            interval = interval_map.get(timeframe, "1min")
            raw = av.intraday(symbol, interval=interval, outputsize="full", cache_ttl=0)
            if raw.empty:
                raise ValueError(f"No data for {symbol}")
            df = raw[(raw.index >= pd.Timestamp(start_s)) & (raw.index <= pd.Timestamp(end_s))]

    elif vendor_l == "yfinance":
        df = _fetch_yfinance(symbol, timeframe, start_s, end_s)

    elif vendor_l == "finnhub":
        key = api_key or os.getenv("FINNHUB_API_KEY", "")
        if not key:
            raise ValueError("FINNHUB_API_KEY not set — add it to .env")
        df = _fetch_finnhub(symbol, timeframe, start_s, end_s, key)

    elif vendor_l in ("stockdata", "stockdataorg"):
        if timeframe != "1D":
            raise ValueError("StockData.org only supports daily (1D) timeframe on free tier")
        key = api_key or os.getenv("STOCKDATA_API_KEY", "")
        if not key:
            raise ValueError("STOCKDATA_API_KEY not set — add it to .env")
        df = _fetch_stockdata(symbol, start_s, end_s, key)

    elif vendor_l in ("massive", "massivecom", "polygon", "polygonio"):
        key = api_key or os.getenv("MASSIVE_API_KEY", "")
        if not key:
            raise ValueError("MASSIVE_API_KEY not set — add it to .env")
        df = _fetch_massive(symbol, timeframe, start_s, end_s, key)

    else:
        raise ValueError(f"Unknown vendor: {vendor!r}. Supported: alphavantage, yfinance, finnhub, stockdata, massive")

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} from {vendor}")

    cache.save(df, vendor, symbol.upper(), timeframe, start_s, end_s)
    return df


def auto_fetch_and_cache(
    symbol: str,
    timeframe: str,
    start: str | date | datetime,
    end: str | date | datetime,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch OHLCV using the best available vendor, automatically.

    Priority (most capable → most reliable fallback):
      1. Massive  — unlimited free, all timeframes (needs MASSIVE_API_KEY)
      2. Finnhub  — 60 req/min, all timeframes   (needs FINNHUB_API_KEY)
      3. StockData — 100 req/day, daily only      (needs STOCKDATA_API_KEY)
      4. yFinance — unlimited, no key required   (always available)

    Returns (DataFrame, vendor_name_used).
    """
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    cache = DataCache()

    # Build priority list — only include keyed vendors if the key exists
    candidates: list[tuple[str, Optional[str]]] = []
    if os.getenv("MASSIVE_API_KEY"):
        candidates.append(("massive", os.getenv("MASSIVE_API_KEY")))
    if os.getenv("FINNHUB_API_KEY"):
        candidates.append(("finnhub", os.getenv("FINNHUB_API_KEY")))
    if os.getenv("STOCKDATA_API_KEY") and timeframe == "1D":
        candidates.append(("stockdata", os.getenv("STOCKDATA_API_KEY")))
    candidates.append(("yfinance", None))  # always last, always works

    last_error: Optional[Exception] = None
    for vendor, key in candidates:
        # Serve from cache first — no API call needed
        existing = cache.load(vendor, symbol.upper(), timeframe, start_s, end_s)
        if existing is not None and not existing.empty:
            return existing, vendor
        try:
            df = fetch_and_cache(vendor, symbol, timeframe, start_s, end_s, api_key=key)
            return df, vendor
        except Exception as exc:
            last_error = exc
            continue

    raise ValueError(
        f"Could not fetch {symbol} from any available vendor. "
        f"Last error: {last_error}"
    )


def get_cached_dataset(
    vendor: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Load from cache only; does not fetch."""
    return DataCache().load(vendor, symbol.upper(), timeframe, start, end)


def list_cached_datasets() -> List[Dict[str, Any]]:
    """List all cached datasets."""
    return DataCache().list_datasets()

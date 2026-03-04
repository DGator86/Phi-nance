"""
phi.data.cache — Dataset Cache Manager
=======================================
Fetch once, store locally, reuse forever.

Storage layout:
  /data_cache/{vendor}/{symbol}/{timeframe}/{start}_{end}.parquet
  /data_cache/{vendor}/{symbol}/{timeframe}/{start}_{end}_metadata.json
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

CACHE_ROOT = Path(__file__).parents[2] / "data_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_dir(vendor: str, symbol: str, timeframe: str) -> Path:
    return CACHE_ROOT / vendor / symbol.upper() / timeframe


def _stem(start, end) -> str:
    s = str(start)[:10].replace("-", "")
    e = str(end)[:10].replace("-", "")
    return f"{s}_{e}"


def get_cache_path(vendor: str, symbol: str, timeframe: str, start, end) -> Path:
    return _cache_dir(vendor, symbol, timeframe) / f"{_stem(start, end)}.parquet"


def get_meta_path(vendor: str, symbol: str, timeframe: str, start, end) -> Path:
    return _cache_dir(vendor, symbol, timeframe) / f"{_stem(start, end)}_metadata.json"


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

        Returns
        -------
        pd.DataFrame or None
        """
        p = self.get_path(vendor, symbol, timeframe, start, end)
        if not p.exists():
            return None
        try:
            if check_staleness:
                meta_p = Path(str(p) + ".metadata.json")
                if meta_p.exists():
                    try:
                        with open(meta_p, encoding="utf-8") as f:
                            meta = json.load(f)
                        fetched_at_str = meta.get("fetched_at")
                        if fetched_at_str:
                            fetched_at = datetime.fromisoformat(fetched_at_str)
                            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - fetched_at).total_seconds() / _SECONDS_PER_DAY
                            is_intraday = timeframe != "1D"
                            max_age = 1 if is_intraday else 7
                            if age_days > max_age:
                                logger.warning(
                                    "Cached data for %s %s is %.1f days old (threshold: %d days).",
                                    symbol, timeframe, age_days, max_age,
                                )
                    except Exception as exc:
                        logger.debug("Cache metadata parse failed for %s: %s", p, exc)
            return pd.read_parquet(p)
        except Exception as exc:
            logger.warning("Cache load failed for %s: %s", p, exc)
            return None

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


def load_metadata(vendor: str, symbol: str, timeframe: str, start, end) -> Optional[Dict[str, Any]]:
    mp = get_meta_path(vendor, symbol, timeframe, start, end)
    if not mp.exists():
        return None
    try:
        with open(mp) as f:
            return json.load(f)
    except Exception:
        return None


def list_cached_datasets() -> List[Dict[str, Any]]:
    """Return metadata dicts for all cached datasets, newest first."""
    results = []
    if not CACHE_ROOT.exists():
        return results
    for meta_file in sorted(CACHE_ROOT.rglob("*_metadata.json")):
        try:
            with open(meta_file) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return sorted(results, key=lambda x: x.get("cached_at", ""), reverse=True)


def delete_dataset(vendor: str, symbol: str, timeframe: str, start, end) -> bool:
    """Delete a cached dataset and its metadata.  Returns True if anything deleted."""
    removed = False
    for p in [
        get_cache_path(vendor, symbol, timeframe, start, end),
        get_meta_path(vendor, symbol, timeframe, start, end),
    ]:
        if p.exists():
            p.unlink()
            removed = True
    return removed


def clear_all_cache() -> int:
    """Delete all cached datasets.  Returns number of parquet files removed."""
    count = 0
    if not CACHE_ROOT.exists():
        return 0
    for p in CACHE_ROOT.rglob("*.parquet"):
        p.unlink()
        count += 1
    for p in CACHE_ROOT.rglob("*_metadata.json"):
        p.unlink()
    return count

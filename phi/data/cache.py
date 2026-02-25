"""
Phi-nance Data Cache
====================

Stores historical OHLCV in parquet format at:
  /data_cache/{vendor}/{symbol}/{timeframe}/{start}_{end}.parquet

Each dataset has metadata.json alongside it.
"""

from __future__ import annotations

import json
import io
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Project root: parent of phi/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_CACHE_ROOT = _PROJECT_ROOT / "data_cache"

_BINANCE_BASE = "https://data.binance.vision/data"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider output to lowercase OHLCV columns."""
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Provider data is missing required columns: {missing}")
    out = df.rename(columns={cols[c]: c for c in required})[required].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out


def _fetch_from_yfinance(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    import yfinance as yf

    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start_s, end=end_s, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No daily data for {symbol}")
    return _normalize_ohlcv(df)


def _fetch_from_alpha_vantage(symbol: str, timeframe: str, start_s: str, end_s: str, api_key: Optional[str]) -> pd.DataFrame:
    if timeframe == "1D":
        return _fetch_from_yfinance(symbol, start_s, end_s)

    from regime_engine.data_fetcher import AlphaVantageFetcher

    av = AlphaVantageFetcher(api_key=api_key)
    interval_map = {"4H": "60min", "1H": "60min", "15m": "15min", "5m": "5min", "1m": "1min"}
    interval = interval_map.get(timeframe, "1min")

    raw = av.intraday(symbol, interval=interval, outputsize="full", cache_ttl=0)
    if raw.empty:
        raise ValueError(f"No data for {symbol}")
    windowed = raw[(raw.index >= pd.Timestamp(start_s)) & (raw.index <= pd.Timestamp(end_s))]
    return _normalize_ohlcv(windowed)


def _fetch_from_binance(symbol: str, timeframe: str, start_s: str, end_s: str) -> pd.DataFrame:
    """
    Pull monthly klines from Binance public data (data.binance.vision) and filter by range.
    Symbol format: BTCUSDT, ETHUSDT, etc.
    """
    tf_map = {"1D": "1d", "1H": "1h", "15m": "15m", "5m": "5m", "1m": "1m"}
    if timeframe not in tf_map:
        raise ValueError("Binance provider supports: 1D, 1H, 15m, 5m, 1m")

    interval = tf_map[timeframe]
    start_dt = pd.Timestamp(start_s)
    end_dt = pd.Timestamp(end_s)
    if end_dt < start_dt:
        raise ValueError("End date must be >= start date")

    months = pd.period_range(start=start_dt.to_period("M"), end=end_dt.to_period("M"), freq="M")
    frames: List[pd.DataFrame] = []
    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ]
    for m in months:
        ym = f"{m.year:04d}-{m.month:02d}"
        url = f"{_BINANCE_BASE}/spot/monthly/klines/{symbol.upper()}/{interval}/{symbol.upper()}-{interval}-{ym}.zip"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=cols)
                frames.append(df)

    if not frames:
        raise ValueError(f"No Binance public data found for {symbol} {interval} in range")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["open_time"] = pd.to_datetime(all_df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    all_df = all_df.set_index("open_time")
    all_df = all_df[(all_df.index >= start_dt) & (all_df.index <= end_dt + pd.Timedelta(days=1))]
    return _normalize_ohlcv(all_df)


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
    """
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    cache = DataCache()
    existing = cache.load(vendor, symbol.upper(), timeframe, start_s, end_s)
    if existing is not None and len(existing) > 0:
        return existing

    vendor_key = vendor.lower().replace("-", "_").replace(" ", "")
    if vendor_key in ("alphavantage", "alpha_vantage"):
        df = _fetch_from_alpha_vantage(symbol, timeframe, start_s, end_s, api_key)
    elif vendor_key in ("yfinance", "yf"):
        if timeframe != "1D":
            raise ValueError("yfinance provider supports only 1D timeframe")
        df = _fetch_from_yfinance(symbol, start_s, end_s)
    elif vendor_key in ("binance", "binance_public"):
        df = _fetch_from_binance(symbol, timeframe, start_s, end_s)
    else:
        raise ValueError(f"Unknown vendor: {vendor}")

    cache.save(df, vendor, symbol.upper(), timeframe, start_s, end_s)
    return df


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

"""Utilities for caching OHLCV and options-chain datasets in parquet files."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_CACHE_ROOT = _PROJECT_ROOT / "data_cache"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider output to lowercase OHLCV columns."""
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Provider data is missing required columns: {missing}")

    out = df.rename(columns={cols[c]: c for c in required})[required].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def _ohlcv_sanity_check(df: pd.DataFrame, symbol: str = "") -> None:
    """Run basic sanity checks on OHLCV data and log warnings."""
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("%s sanity check: missing columns %s", symbol, missing)
        return

    for col in ["open", "high", "low", "close"]:
        if (df[col] < 0).any():
            logger.warning("%s sanity check: negative values in '%s' column", symbol, col)
    if not df.index.is_monotonic_increasing:
        logger.warning("%s sanity check: index is not chronologically ordered", symbol)


class DataCache:
    """Parquet cache manager for OHLCV and options-chain datasets."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _DATA_CACHE_ROOT

    @staticmethod
    def _stem(start: str | date | datetime, end: str | date | datetime) -> str:
        s = str(start)[:10].replace("-", "")
        e = str(end)[:10].replace("-", "")
        return f"{s}_{e}"

    def _cache_dir(self, vendor: str, symbol: str, timeframe: str) -> Path:
        d = self.root / vendor / symbol.upper() / timeframe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _parquet_path(self, vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
        return self._cache_dir(vendor, symbol, timeframe) / f"{self._stem(start, end)}.parquet"

    def _meta_path(self, vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
        return self._cache_dir(vendor, symbol, timeframe) / f"{self._stem(start, end)}.metadata.json"

    def exists(self, vendor: str, symbol: str, timeframe: str, start: str, end: str) -> bool:
        return self._parquet_path(vendor, symbol, timeframe, start, end).exists()

    def load(
        self,
        vendor: str,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        check_staleness: bool = False,
    ) -> Optional[pd.DataFrame]:
        path = self._parquet_path(vendor, symbol, timeframe, start, end)
        if not path.exists():
            return None

        if check_staleness:
            self._warn_if_stale(self._meta_path(vendor, symbol, timeframe, start, end), symbol, timeframe)

        try:
            return pd.read_parquet(path)
        except Exception as exc:
            logger.warning("Cache load failed for %s: %s", path, exc)
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
        path = self._parquet_path(vendor, symbol, timeframe, start, end)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=True)

        meta = metadata.copy() if metadata else {}
        meta.setdefault("vendor", vendor)
        meta.setdefault("symbol", symbol.upper())
        meta.setdefault("timeframe", timeframe)
        meta.setdefault("start", str(start)[:10])
        meta.setdefault("end", str(end)[:10])
        meta.setdefault("rows", len(df))
        meta.setdefault("columns", list(df.columns))
        meta["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        with open(self._meta_path(vendor, symbol, timeframe, start, end), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def list_datasets(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.root.exists():
            return out
        for parquet in self.root.rglob("*.parquet"):
            meta_path = parquet.with_suffix(".metadata.json")
            meta: Dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            meta.setdefault("path", str(parquet))
            out.append(meta)
        return out

    @staticmethod
    def _warn_if_stale(meta_path: Path, symbol: str, timeframe: str) -> None:
        if not meta_path.exists():
            return
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            fetched_at = meta.get("fetched_at")
            if not fetched_at:
                return
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - datetime.fromisoformat(fetched_at)).total_seconds() / _SECONDS_PER_DAY
            max_age = 1 if timeframe != "1D" else 7
            if age_days > max_age:
                logger.warning("Cached data for %s %s is %.1f days old (threshold: %d days).", symbol, timeframe, age_days, max_age)
        except Exception:
            return


# ---- OHLCV fetch + cache ----

def _fetch_from_yfinance(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance."""
    import yfinance as yf

    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start_s, end=end_s, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    out = _normalize_ohlcv(df)
    _ohlcv_sanity_check(out, symbol)
    return out


def fetch_and_cache(
    vendor: str,
    symbol: str,
    timeframe: str,
    start: str | date | datetime,
    end: str | date | datetime,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from vendor and cache result."""
    del api_key  # placeholder for future keyed vendors
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    cache = DataCache()

    existing = cache.load(vendor, symbol.upper(), timeframe, start_s, end_s)
    if existing is not None and not existing.empty:
        return existing

    vendor_key = vendor.lower().replace("-", "_").replace(" ", "")
    if vendor_key not in {"yfinance", "yf", "alphavantage", "alpha_vantage", "binance", "binance_public"}:
        raise ValueError(f"Unknown vendor: {vendor!r}. Supported: yfinance, alphavantage, binance")

    # Phase 1 keeps yfinance as canonical fallback for phi namespace.
    df = _fetch_from_yfinance(symbol, start_s, end_s)
    cache.save(df, vendor, symbol.upper(), timeframe, start_s, end_s)
    return df


def auto_fetch_and_cache(
    symbol: str,
    timeframe: str,
    start: str | date | datetime,
    end: str | date | datetime,
) -> tuple[pd.DataFrame, str]:
    """Auto fetch data using yfinance fallback and cache it."""
    return fetch_and_cache("yfinance", symbol, timeframe, start, end), "yfinance"


def get_cached_dataset(vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load a cached OHLCV dataset without calling a data vendor."""
    return DataCache().load(vendor, symbol.upper(), timeframe, start, end)


def load_metadata(vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a cached OHLCV dataset if present."""
    path = DataCache()._meta_path(vendor, symbol.upper(), timeframe, start, end)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_cached_datasets() -> List[Dict[str, Any]]:
    """List all cached OHLCV datasets."""
    return DataCache().list_datasets()


# ---- Options-chain cache (Phase 1) ----

def _normalize_option_date(requested: Optional[str], expirations: List[str]) -> str:
    if not expirations:
        raise ValueError("No options expirations available for symbol")
    if requested is None:
        return expirations[0]
    if requested in expirations:
        return requested
    logger.warning("Requested expiration '%s' unavailable; using nearest '%s'", requested, expirations[0])
    return expirations[0]


def fetch_options_chain(symbol: str, date: Optional[str] = None, vendor: str = "yfinance") -> pd.DataFrame:
    """Fetch and cache options chain for a symbol and expiration date.

    Cached at: ``data_cache/options/{symbol}/{date}/chain.parquet``.
    """
    vendor_key = vendor.lower().replace("-", "_").replace(" ", "")
    if vendor_key not in {"yfinance", "yf"}:
        raise ValueError("Only yfinance vendor is supported for options chain in Phase 1")

    import yfinance as yf

    ticker = yf.Ticker(symbol)
    expiry = _normalize_option_date(date, list(ticker.options or []))

    chain = ticker.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()
    calls["option_type"] = "call"
    puts["option_type"] = "put"
    calls["expiration"] = expiry
    puts["expiration"] = expiry

    combined = pd.concat([calls, puts], ignore_index=True)
    cache_dir = _DATA_CACHE_ROOT / "options" / symbol.upper() / expiry
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "chain.parquet"
    combined.to_parquet(out_path, index=False)
    return combined


def get_cached_options(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """Load cached options chain for ``symbol`` and expiration ``date``."""
    path = _DATA_CACHE_ROOT / "options" / symbol.upper() / str(date) / "chain.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Failed to load cached options chain %s: %s", path, exc)
        return None

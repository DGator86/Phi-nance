"""Utilities for caching OHLCV and options-chain datasets in parquet files."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import fasteners
import pandas as pd
import pandera as pa
import requests
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import logging
from phi.logging import get_logger

logger = get_logger(__name__)

DATA_CACHE_ROOT = Path(os.getenv("DATA_CACHE_ROOT", "./data_cache"))
DATA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
# Compatibility alias used by other modules.
_DATA_CACHE_ROOT = DATA_CACHE_ROOT


class DataFetchError(RuntimeError):
    """Raised when vendor fetch or cache persistence fails."""


class CacheCorruptedError(DataFetchError):
    """Raised when cached parquet/metadata payload is missing or malformed."""


OHLCV_SCHEMA = pa.DataFrameSchema(
    {
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "volume": pa.Column(float, nullable=False),
    },
    index=pa.Index(pa.DateTime, nullable=False),
    coerce=True,
)

_TIMEFRAME_MAX_AGE = {
    "1D": timedelta(hours=24),
    "1H": timedelta(hours=1),
    "1M": timedelta(minutes=1),
}
_DEFAULT_MAX_AGE = timedelta(days=7)
_LOCK_TIMEOUT_SECONDS = 30


def _meta_path_from_cache(cache_path: Path) -> Path:
    """Return the sidecar metadata path for a parquet cache file."""
    return cache_path.with_suffix(".meta.json")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider output to lowercase OHLCV columns."""
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"Provider data is missing required columns: {missing}")

    out = df.rename(columns={cols[c]: c for c in required})[required].copy()
    out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def _validate_ohlcv(df: pd.DataFrame, symbol: str, vendor: str) -> pd.DataFrame:
    """Validate OHLCV dataframe shape and types."""
    try:
        validated = OHLCV_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        logger.error("OHLCV validation failed for %s/%s: %s", vendor, symbol, exc)
        raise ValueError(f"OHLCV validation failed for {vendor}/{symbol}: {exc}") from exc
    logger.info("Validation succeeded for %s/%s (%d rows)", vendor, symbol, len(validated))
    return validated


def _hash_dataframe(df: pd.DataFrame) -> str:
    """Compute stable SHA-256 hash for dataframe contents and index."""
    row_hashes = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()


@contextmanager
def _file_lock(lock_path: Path, timeout: int = _LOCK_TIMEOUT_SECONDS) -> Generator[None, None, None]:
    """Acquire and release an inter-process file lock."""
    lock = fasteners.InterProcessLock(str(lock_path))
    logger.info("Attempting lock acquisition: %s", lock_path)
    acquired = lock.acquire(blocking=True, timeout=timeout)
    if not acquired:
        raise TimeoutError(f"Could not acquire cache lock within {timeout}s: {lock_path}")
    logger.info("Acquired lock: %s", lock_path)
    try:
        yield
    finally:
        lock.release()
        logger.info("Released lock: %s", lock_path)


class DataCache:
    """Parquet cache manager for OHLCV and options-chain datasets."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or DATA_CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

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
        return _meta_path_from_cache(self._parquet_path(vendor, symbol, timeframe, start, end))

    def exists(self, vendor: str, symbol: str, timeframe: str, start: str, end: str) -> bool:
        parquet = self._parquet_path(vendor, symbol, timeframe, start, end)
        meta = self._meta_path(vendor, symbol, timeframe, start, end)
        return parquet.exists() and meta.exists()

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
        meta_path = self._meta_path(vendor, symbol, timeframe, start, end)
        if not path.exists():
            logger.info("Cache miss: %s", path)
            return None
        if not meta_path.exists():
            logger.warning("Cache metadata missing for %s; treating as corrupted", path)
            return None

        if check_staleness and is_cache_stale(path, timeframe):
            logger.info("Cache stale for %s", path)
            return None

        try:
            logger.info("Cache hit: %s", path)
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
            logger.warning("Cache load failed for %s: %s", path, exc)
            raise CacheCorruptedError(f"Cache read failed for {path}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while loading cache %s", path)
            raise CacheCorruptedError(f"Unexpected cache read error for {path}: {exc}") from exc

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
        meta_path = _meta_path_from_cache(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")

        meta = metadata.copy() if metadata else {}
        meta.update(
            {
                "vendor": vendor,
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "start": str(start)[:10],
                "end": str(end)[:10],
                "fetch_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data_hash": _hash_dataframe(df),
            }
        )

        try:
            with _file_lock(lock_path):
                df.to_parquet(path, index=True)
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except (TimeoutError, OSError, ValueError, TypeError) as exc:
            logger.error("Failed to persist cache dataset %s: %s", path, exc)
            raise DataFetchError(f"Failed to persist cache dataset {path}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected cache persistence error for %s", path)
            raise DataFetchError(f"Unexpected cache persistence error for {path}: {exc}") from exc

        logger.info("Saved cache dataset %s and metadata %s", path, meta_path)
        return path

    def list_datasets(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.root.exists():
            return out
        for parquet in self.root.rglob("*.parquet"):
            meta_path = _meta_path_from_cache(parquet)
            meta: Dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, JSONDecodeError, TypeError):
                    meta = {}
            meta.setdefault("path", str(parquet))
            out.append(meta)
        return out


def _parse_fetch_timestamp(raw_ts: str) -> datetime:
    """Parse sidecar metadata fetch timestamp."""
    return datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))


def is_cache_stale(cache_path: Path, timeframe: str, max_age_hours: Optional[float] = None) -> bool:
    """Return whether cache should be considered stale using metadata fetch timestamp."""
    meta_path = _meta_path_from_cache(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        return True

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        fetch_ts = _parse_fetch_timestamp(metadata["fetch_timestamp"])
    except (OSError, JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Invalid metadata at %s: %s", meta_path, exc)
        return True

    age_limit = timedelta(hours=max_age_hours) if max_age_hours is not None else _TIMEFRAME_MAX_AGE.get(
        timeframe.upper(), _DEFAULT_MAX_AGE
    )
    age = datetime.now(timezone.utc) - fetch_ts
    stale = age > age_limit
    if stale:
        logger.info("Cache stale check: %s age=%s limit=%s", cache_path, age, age_limit)
    return stale


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(requests.RequestException) | retry_if_exception_type(ConnectionError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_with_retry(fetcher: Any, symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Invoke a fetcher with retry behavior for transient network failures."""
    return fetcher(symbol, start_s, end_s)


# ---- OHLCV fetch + cache ----

def _fetch_from_yfinance(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    import yfinance as yf

    logger.info("Fetching yfinance data for %s (%s to %s)", symbol, start_s, end_s)
    tkr = yf.Ticker(symbol)
    df = tkr.history(start=start_s, end=end_s, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    return _normalize_ohlcv(df)


def _fetch_from_alphavantage(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Phase-1 AlphaVantage handler (delegates to yfinance fallback)."""
    logger.warning("AlphaVantage fetch fallback active for %s; using yfinance", symbol)
    return _fetch_from_yfinance(symbol, start_s, end_s)


def _get_fetcher(vendor: str) -> Any:
    """Resolve vendor key to fetcher function."""
    vendor_key = vendor.lower().replace("-", "_").replace(" ", "")
    fetchers = {
        "yfinance": _fetch_from_yfinance,
        "yf": _fetch_from_yfinance,
        "alphavantage": _fetch_from_alphavantage,
        "alpha_vantage": _fetch_from_alphavantage,
    }
    if vendor_key not in fetchers:
        raise ValueError(f"Unknown vendor: {vendor!r}. Supported: {', '.join(sorted(fetchers))}")
    return fetchers[vendor_key]


def fetch_and_cache(
    vendor: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV from vendor and persist a validated cache dataset."""
    symbol_u = symbol.upper()
    start_s = str(start)[:10]
    end_s = str(end)[:10]
    cache = DataCache()
    cache_path = cache._parquet_path(vendor, symbol_u, timeframe, start_s, end_s)
    meta_path = _meta_path_from_cache(cache_path)

    if cache_path.exists() and not force_refresh:
        if not meta_path.exists():
            logger.warning("Metadata sidecar missing for %s; cache considered corrupted", cache_path)
        elif not is_cache_stale(cache_path, timeframe):
            try:
                cached = cache.load(vendor, symbol_u, timeframe, start_s, end_s)
                if cached is not None:
                    return cached
            except CacheCorruptedError as exc:
                logger.warning("Corrupted cache detected for %s/%s, forcing refresh: %s", vendor, symbol_u, exc)

    fetcher = _get_fetcher(vendor)
    logger.info("Cache miss/stale; fetching %s %s %s %s-%s", vendor, symbol_u, timeframe, start_s, end_s)

    try:
        fetched = _fetch_with_retry(fetcher, symbol_u, start_s, end_s)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to fetch data for %s/%s", vendor, symbol_u)
        raise DataFetchError(f"Failed to fetch data for {vendor}/{symbol_u} after retries: {exc}") from exc

    validated = _validate_ohlcv(fetched, symbol_u, vendor)

    metadata = {
        "vendor": vendor,
        "symbol": symbol_u,
        "timeframe": timeframe,
        "start": start_s,
        "end": end_s,
    }
    try:
        cache.save(validated, vendor, symbol_u, timeframe, start_s, end_s, metadata=metadata)
    except DataFetchError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected cache save failure for %s/%s", vendor, symbol_u)
        raise DataFetchError(f"Failed to write cache for {vendor}/{symbol_u}: {exc}") from exc

    return validated


def auto_fetch_and_cache(
    symbol: str,
    timeframe: str,
    start: str | date | datetime,
    end: str | date | datetime,
) -> tuple[pd.DataFrame, str]:
    """Auto fetch data using yfinance fallback and cache it."""
    return fetch_and_cache("yfinance", symbol, timeframe, str(start), str(end)), "yfinance"


def get_cached_dataset(vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load a cached OHLCV dataset without calling a data vendor."""
    try:
        return DataCache().load(vendor, symbol.upper(), timeframe, start, end)
    except CacheCorruptedError as exc:
        logger.warning("Cached dataset is corrupted for %s/%s: %s", vendor, symbol.upper(), exc)
        return None


def load_metadata(vendor: str, symbol: str, timeframe: str, start: str, end: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a cached OHLCV dataset if present."""
    path = DataCache()._meta_path(vendor, symbol.upper(), timeframe, start, end)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError, TypeError):
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
    cache_dir = DATA_CACHE_ROOT / "options" / symbol.upper() / expiry
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "chain.parquet"
    combined.to_parquet(out_path, index=False)
    return combined


def get_cached_options(symbol: str, date: str) -> Optional[pd.DataFrame]:
    """Load cached options chain for ``symbol`` and expiration ``date``."""
    path = DATA_CACHE_ROOT / "options" / symbol.upper() / str(date) / "chain.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        logger.warning("Failed to load cached options chain %s: %s", path, exc)
        return None

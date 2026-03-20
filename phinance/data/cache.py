"""
phinance.data.cache
===================

Parquet-based OHLCV dataset cache with vendor-agnostic fetch_and_cache().

Storage layout
--------------
  data_cache/
    {vendor}/
      {SYMBOL}/
        {timeframe}/
          {start}_{end}.parquet
          {start}_{end}.parquet.metadata.json

Public API
----------
  DataCache                — Class: save / load / list cached datasets
  fetch_and_cache(...)     — Fetch from vendor (or return cache hit)
  get_cached_dataset(...)  — Load from cache only; no network call
  list_cached_datasets()   — List all cached datasets with metadata
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from phinance.utils.logging import get_logger
from phinance.data.optimised_cache import OptimisedCache

logger = get_logger(__name__)

_SECONDS_PER_DAY = 86_400
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_CACHE_ROOT = _PROJECT_ROOT / "data_cache"
_OPTIMISED_CACHE: OptimisedCache | None = None


# ── Normalisation + Validation ────────────────────────────────────────────────


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise provider output to lowercase OHLCV columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw provider DataFrame (any column casing).

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with columns ``[open, high, low, close, volume]``
        and a tz-naive DatetimeIndex sorted ascending.

    Raises
    ------
    phinance.exceptions.DataValidationError
        When any required column is missing.
    """
    from phinance.exceptions import DataValidationError

    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise DataValidationError(
            f"Provider data is missing required columns: {missing}"
        )
    out = df.rename(columns={cols[c]: c for c in required})[required].copy()
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    return out


def ohlcv_sanity_check(df: pd.DataFrame, symbol: str = "") -> None:
    """Run basic sanity checks on an OHLCV DataFrame (logs warnings only).

    Checks
    ------
    - All required columns present.
    - No negative prices in open/high/low/close.
    - Index is monotonically increasing.
    """
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("%s sanity: missing columns %s", symbol, missing)
        return
    for col in ["open", "high", "low", "close"]:
        if (df[col] < 0).any():
            logger.warning(
                "%s sanity: negative values found in '%s' column", symbol, col
            )
    if not df.index.is_monotonic_increasing:
        logger.warning(
            "%s sanity: index is not chronologically ordered", symbol
        )


# ── DataCache ─────────────────────────────────────────────────────────────────


class DataCache:
    """Manages a parquet-based OHLCV cache rooted at ``data_cache/``.

    Parameters
    ----------
    root : Path, optional
        Root directory for the cache.  Defaults to ``<project_root>/data_cache``.

    Example
    -------
        cache = DataCache()
        df = cache.load("yfinance", "SPY", "1D", "2023-01-01", "2023-12-31")
        if df is None:
            df = fetch_from_somewhere(...)
            cache.save(df, "yfinance", "SPY", "1D", "2023-01-01", "2023-12-31")
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root: Path = root or _DATA_CACHE_ROOT

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _parquet_path(
        self, vendor: str, symbol: str, timeframe: str, start: str, end: str
    ) -> Path:
        s = str(start)[:10].replace("-", "")
        e = str(end)[:10].replace("-", "")
        base = self.root / vendor / symbol.upper() / timeframe
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{s}_{e}.parquet"

    def _meta_path(
        self, vendor: str, symbol: str, timeframe: str, start: str, end: str
    ) -> Path:
        return Path(
            str(self._parquet_path(vendor, symbol, timeframe, start, end))
            + ".metadata.json"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def exists(
        self, vendor: str, symbol: str, timeframe: str, start: str, end: str
    ) -> bool:
        """Return True if a cached parquet file exists for this key."""
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
        """Load cached dataset if it exists.

        Parameters
        ----------
        check_staleness : bool
            When *True*, warn if cached data is older than 1 day (intraday)
            or 7 days (daily).

        Returns
        -------
        pd.DataFrame or None
        """
        p = self._parquet_path(vendor, symbol, timeframe, start, end)
        if not p.exists():
            return None
        try:
            if check_staleness:
                self._warn_if_stale(
                    self._meta_path(vendor, symbol, timeframe, start, end),
                    symbol, timeframe,
                )
            return pd.read_parquet(p)
        except Exception as exc:
            logger.warning("Cache load failed for %s %s: %s", symbol, timeframe, exc)
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
        """Persist a DataFrame and its metadata to disk.

        Returns
        -------
        Path
            Location of the saved parquet file.
        """
        p = self._parquet_path(vendor, symbol, timeframe, start, end)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=True)

        meta: Dict[str, Any] = metadata or {}
        meta.setdefault("vendor", vendor)
        meta.setdefault("symbol", symbol)
        meta.setdefault("timeframe", timeframe)
        meta.setdefault("start", str(start)[:10])
        meta.setdefault("end", str(end)[:10])
        meta.setdefault("rows", len(df))
        meta.setdefault("columns", list(df.columns))
        meta["fetched_at"] = (
            datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        )
        meta_p = self._meta_path(vendor, symbol, timeframe, start, end)
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.debug(
            "Cached %d rows → %s", len(df), p.relative_to(self.root)
        )
        return p

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return metadata dicts for every cached parquet file."""
        results: List[Dict[str, Any]] = []
        if not self.root.exists():
            return results
        for parquet in self.root.rglob("*.parquet"):
            meta_path = Path(str(parquet) + ".metadata.json")
            meta: Dict[str, Any] = {}
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
            meta["path"] = str(parquet)
            results.append(meta)
        return results

    # ── Staleness helper ──────────────────────────────────────────────────────

    @staticmethod
    def _warn_if_stale(meta_path: Path, symbol: str, timeframe: str) -> None:
        if not meta_path.exists():
            return
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            fetched_str = meta.get("fetched_at")
            if not fetched_str:
                return
            fetched = datetime.fromisoformat(fetched_str)
            age_days = (
                datetime.now(timezone.utc).replace(tzinfo=None) - fetched
            ).total_seconds() / _SECONDS_PER_DAY
            max_age = 1 if timeframe != "1D" else 7
            if age_days > max_age:
                logger.warning(
                    "Cached data for %s %s is %.1f days old (threshold: %d).",
                    symbol, timeframe, age_days, max_age,
                )
        except Exception:
            pass


# ── Top-level convenience functions ──────────────────────────────────────────


def fetch_and_cache(
    vendor: str,
    symbol: str,
    timeframe: str,
    start: Union[str, date, datetime],
    end: Union[str, date, datetime],
    api_key: Optional[str] = None,
    use_optimised_cache: bool = False,
    optimised_cache: OptimisedCache | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from a vendor and persist it; return cache hit if available.

    Parameters
    ----------
    vendor : str
        One of ``"yfinance"``, ``"alphavantage"``, ``"binance"``.
    symbol : str
        Ticker / pair (e.g. ``"SPY"``, ``"BTCUSDT"``).
    timeframe : str
        One of ``"1D"``, ``"1H"``, ``"15m"``, ``"5m"``, ``"1m"``.
    start, end : str | date | datetime
        Date range.
    api_key : str, optional
        API key for vendors that require one (Alpha Vantage).

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame.

    Raises
    ------
    phinance.exceptions.UnsupportedVendorError
        When the vendor string is not recognised.
    phinance.exceptions.DataFetchError
        When the vendor fetch fails after all retries.
    """
    from phinance.data.vendors.yfinance import YFinanceVendor
    from phinance.data.vendors.alphavantage import AlphaVantageVendor
    from phinance.data.vendors.binance import BinanceVendor
    from phinance.exceptions import UnsupportedVendorError

    start_s = str(start)[:10]
    end_s = str(end)[:10]

    cache = DataCache()
    active_optimised_cache = optimised_cache if optimised_cache is not None else _get_global_optimised_cache(use_optimised_cache)

    cache_key = (vendor, symbol.upper(), timeframe, start_s, end_s)
    cached = active_optimised_cache.get(cache_key) if active_optimised_cache is not None else None
    if cached is None:
        cached = cache.load(vendor, symbol.upper(), timeframe, start_s, end_s)
        if cached is not None and active_optimised_cache is not None:
            active_optimised_cache.set(cache_key, cached)
    if cached is not None and len(cached) > 0:
        logger.debug("Cache hit: %s %s %s", vendor, symbol, timeframe)
        return cached

    vendor_key = vendor.lower().replace("-", "_").replace(" ", "")
    if vendor_key in ("yfinance", "yf"):
        v = YFinanceVendor()
    elif vendor_key in ("alphavantage", "alpha_vantage"):
        v = AlphaVantageVendor(api_key=api_key)
    elif vendor_key in ("binance", "binance_public"):
        v = BinanceVendor()
    else:
        raise UnsupportedVendorError(
            f"Unknown vendor: '{vendor}'. Supported: yfinance, alphavantage, binance."
        )

    df = v.fetch(symbol=symbol, timeframe=timeframe, start=start_s, end=end_s)
    cache.save(df, vendor, symbol.upper(), timeframe, start_s, end_s)
    if active_optimised_cache is not None:
        active_optimised_cache.set(cache_key, df)
    return df


def _get_global_optimised_cache(enabled: bool) -> OptimisedCache | None:
    global _OPTIMISED_CACHE
    if not enabled:
        return None
    if _OPTIMISED_CACHE is None:
        _OPTIMISED_CACHE = OptimisedCache(max_size_mb=1024, default_ttl_seconds=300)
    return _OPTIMISED_CACHE


def get_cached_dataset(
    vendor: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Load from cache only — does not trigger a network fetch."""
    return DataCache().load(vendor, symbol.upper(), timeframe, start, end)


def list_cached_datasets() -> List[Dict[str, Any]]:
    """Return a list of metadata dicts for all cached datasets."""
    return DataCache().list_datasets()

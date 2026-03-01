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
import logging
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

_SECONDS_PER_DAY = 86400

logger = logging.getLogger(__name__)

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


def _ohlcv_sanity_check(df: pd.DataFrame, symbol: str = "") -> None:
    """Run basic sanity checks on OHLCV data and log warnings.

    Checks performed:
    - All OHLCV columns present.
    - No negative prices in open, high, low, close.
    - Index is monotonically increasing (chronological).

    Parameters
    ----------
    df : pd.DataFrame
        Normalised OHLCV DataFrame.
    symbol : str, optional
        Symbol name used in log messages.
    """
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


def _fetch_from_yfinance(symbol: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance with exponential-backoff retry.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"SPY"``).
    start_s : str
        Start date in ``YYYY-MM-DD`` format.
    end_s : str
        End date in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame.

    Raises
    ------
    ValueError
        When no data is returned after all retries.
    """
    import yfinance as yf

    last_exc: Exception = ValueError(f"No daily data for {symbol}")
    for attempt in range(3):
        try:
            tkr = yf.Ticker(symbol)
            df = tkr.history(start=start_s, end=end_s, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No daily data for {symbol}")
            result = _normalize_ohlcv(df)
            _ohlcv_sanity_check(result, symbol)
            logger.info("yfinance: fetched %d rows for %s", len(result), symbol)
            return result
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("yfinance attempt %d/%d failed for %s: %s. Retrying in %ds.", attempt + 1, 3, symbol, exc, wait)
            if attempt < 2:
                time.sleep(wait)
    raise last_exc


# yfinance intraday interval map + lookback caps (in calendar days)
_YF_INTRADAY_MAP = {
    "1m": ("1m", 7),
    "5m": ("5m", 60),
    "15m": ("15m", 60),
    "30m": ("30m", 60),
    "1H": ("1h", 730),
}


def _fetch_from_yfinance_intraday(symbol: str, timeframe: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch intraday OHLCV from yfinance (no API key, no rate limit).

    yfinance supports intraday via the ``interval`` parameter. Maximum
    historical depth depends on the interval:

    - ``1m``  → last 7 calendar days
    - ``5m``  → last 60 calendar days
    - ``15m`` → last 60 calendar days
    - ``30m`` → last 60 calendar days
    - ``1H``  → last 730 calendar days

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. ``"SPY"``, ``"AAPL"``).
    timeframe : str
        One of ``"1m"``, ``"5m"``, ``"15m"``, ``"30m"``, ``"1H"``.
    start_s : str
        Start date ``YYYY-MM-DD``.
    end_s : str
        End date ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame.

    Raises
    ------
    ValueError
        When the timeframe is unsupported or no data is returned.
    """
    import yfinance as yf

    if timeframe not in _YF_INTRADAY_MAP:
        raise ValueError(
            f"yfinance intraday supports: {list(_YF_INTRADAY_MAP)}. Got: {timeframe}"
        )
    interval, max_days = _YF_INTRADAY_MAP[timeframe]

    # Warn if requested range exceeds yfinance lookback cap.
    start_dt = pd.Timestamp(start_s)
    cap_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=max_days)
    if start_dt < cap_start:
        logger.warning(
            "yfinance %s only keeps %d days of history. "
            "Clamping start from %s to %s.",
            interval, max_days, start_s, cap_start.date(),
        )
        start_s = str(cap_start.date())

    last_exc: Exception = ValueError(f"No intraday data for {symbol}")
    for attempt in range(3):
        try:
            tkr = yf.Ticker(symbol)
            df = tkr.history(start=start_s, end=end_s, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No intraday data for {symbol} {interval}")
            # tz-strip so index is tz-naive (consistent with other providers)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            result = _normalize_ohlcv(df)
            _ohlcv_sanity_check(result, symbol)
            logger.info("yfinance intraday: fetched %d rows for %s %s", len(result), symbol, interval)
            return result
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "yfinance intraday attempt %d/%d failed for %s %s: %s. Retrying in %ds.",
                attempt + 1, 3, symbol, interval, exc, wait,
            )
            if attempt < 2:
                time.sleep(wait)
    raise last_exc


def _fetch_from_alpha_vantage(symbol: str, timeframe: str, start_s: str, end_s: str, api_key: Optional[str]) -> pd.DataFrame:
    """Fetch OHLCV from Alpha Vantage with exponential-backoff retry.

    For daily timeframe delegates to :func:`_fetch_from_yfinance`.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    timeframe : str
        One of ``"1D"``, ``"4H"``, ``"1H"``, ``"15m"``, ``"5m"``, ``"1m"``.
    start_s : str
        Start date ``YYYY-MM-DD``.
    end_s : str
        End date ``YYYY-MM-DD``.
    api_key : str, optional
        Alpha Vantage API key.

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame.
    """
    if timeframe == "1D":
        return _fetch_from_yfinance(symbol, start_s, end_s)

    from regime_engine.data_fetcher import AlphaVantageFetcher

    last_exc: Exception = ValueError(f"No data for {symbol}")
    for attempt in range(3):
        try:
            av = AlphaVantageFetcher(api_key=api_key)
            interval_map = {"4H": "60min", "1H": "60min", "15m": "15min", "5m": "5min", "1m": "1min"}
            interval = interval_map.get(timeframe, "1min")
            raw = av.intraday(symbol, interval=interval, outputsize="full", cache_ttl=0)
            if raw.empty:
                raise ValueError(f"No data for {symbol}")
            windowed = raw[(raw.index >= pd.Timestamp(start_s)) & (raw.index <= pd.Timestamp(end_s))]
            result = _normalize_ohlcv(windowed)
            _ohlcv_sanity_check(result, symbol)
            logger.info("alpha_vantage: fetched %d rows for %s", len(result), symbol)
            return result
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("alpha_vantage attempt %d/%d failed for %s: %s. Retrying in %ds.", attempt + 1, 3, symbol, exc, wait)
            if attempt < 2:
                time.sleep(wait)
    raise last_exc


def _fetch_from_binance(symbol: str, timeframe: str, start_s: str, end_s: str) -> pd.DataFrame:
    """Fetch OHLCV from Binance public data with exponential-backoff retry.

    Downloads monthly kline zip files from ``data.binance.vision`` and
    filters the result to the requested date range.

    Parameters
    ----------
    symbol : str
        Binance pair symbol (e.g. ``"BTCUSDT"``).
    timeframe : str
        One of ``"1D"``, ``"1H"``, ``"15m"``, ``"5m"``, ``"1m"``.
    start_s : str
        Start date ``YYYY-MM-DD``.
    end_s : str
        End date ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        Normalised OHLCV DataFrame.

    Raises
    ------
    ValueError
        When no data is found or the timeframe is unsupported.
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
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, header=None, names=cols)
                        frames.append(df)
                logger.info("binance: fetched %s %s %s", symbol, interval, ym)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning("binance attempt %d/%d failed for %s %s: %s. Retrying in %ds.", attempt + 1, 3, symbol, ym, exc, wait)
                if attempt < 2:
                    time.sleep(wait)
        else:
            logger.warning("binance: all retries exhausted for %s %s", symbol, ym)

    if not frames:
        raise ValueError(f"No Binance public data found for {symbol} {interval} in range")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["open_time"] = pd.to_datetime(all_df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    all_df = all_df.set_index("open_time")
    all_df = all_df[(all_df.index >= start_dt) & (all_df.index <= end_dt + pd.Timedelta(days=1))]
    result = _normalize_ohlcv(all_df)
    _ohlcv_sanity_check(result, symbol)
    return result


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
        s = str(start)[:10].replace("-", "")
        e = str(end)[:10].replace("-", "")
        base = self.root / vendor / symbol.upper() / timeframe
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{s}_{e}.parquet"

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
        check_staleness: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Load cached dataset if it exists.

        Parameters
        ----------
        vendor, symbol, timeframe, start, end : str
            Dataset identifiers.
        check_staleness : bool
            When ``True``, warn if cached data is older than 1 day for intraday
            timeframes or 7 days for daily timeframes.

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
                    except Exception:
                        pass
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
        meta["fetched_at"] = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        meta_path = Path(str(p) + ".metadata.json")
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
        if timeframe == "1D":
            df = _fetch_from_yfinance(symbol, start_s, end_s)
        else:
            # No API key needed, no rate limiting — yfinance intraday via interval param.
            df = _fetch_from_yfinance_intraday(symbol, timeframe, start_s, end_s)
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

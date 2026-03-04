"""
Polygon backfill for 1m bars — 2+ years. Required for walk-forward; Tradier alone insufficient.

Uses httpx by default. If the official client is installed (pip install massive),
use fetch_1m_bars_massive() for list_aggs-style pagination (port from polygon-io/client-python).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Default store root: data/bars/{ticker}/{year}.parquet
DEFAULT_STORE_ROOT = Path(__file__).resolve().parents[2] / "data" / "bars"


def fetch_1m_bars_massive(
    ticker: str,
    from_ts: datetime,
    to_ts: datetime,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch 1m bars via official Massive (Polygon) client when installed.
    Ported from massive-com/client-python: client.list_aggs(ticker=, multiplier=1, timespan="minute", from_=, to=, limit=50000).
    """
    try:
        from massive import RESTClient
    except ImportError:
        raise ImportError("pip install massive to use fetch_1m_bars_massive")
    key = api_key or os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise ValueError("POLYGON_API_KEY or MASSIVE_API_KEY required")
    client = RESTClient(api_key=key)
    from_str = from_ts.strftime("%Y-%m-%d")
    to_str = to_ts.strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for a in client.list_aggs(
        ticker=ticker.upper(),
        multiplier=1,
        timespan="minute",
        from_=from_str,
        to=to_str,
        limit=50000,
    ):
        t = getattr(a, "timestamp", None) or getattr(a, "t", None)
        if t is None:
            continue
        if hasattr(t, "timestamp"):
            ts = pd.Timestamp(t.timestamp(), unit="s", tz="America/New_York")
        else:
            ts = pd.Timestamp(t, unit="ms", tz="America/New_York")
        rows.append({
            "timestamp": ts,
            "open": getattr(a, "open", a.get("o") if isinstance(a, dict) else 0),
            "high": getattr(a, "high", a.get("h") if isinstance(a, dict) else 0),
            "low": getattr(a, "low", a.get("l") if isinstance(a, dict) else 0),
            "close": getattr(a, "close", a.get("c") if isinstance(a, dict) else 0),
            "volume": getattr(a, "volume", a.get("v", 0) if isinstance(a, dict) else 0),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_1m_bars_polygon(
    ticker: str,
    from_ts: datetime,
    to_ts: datetime,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch 1m aggregates from Polygon.io (REST). Returns DataFrame with columns
    timestamp, open, high, low, close, volume. Use fetch_1m_bars_massive when massive is installed.
    """
    import httpx
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY required for backfill")
    base = "https://api.polygon.io"
    from_str = from_ts.strftime("%Y-%m-%d")
    to_str = to_ts.strftime("%Y-%m-%d")
    url = f"{base}/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{from_str}/{to_str}"
    all_rows: list[dict[str, Any]] = []
    while url:
        with httpx.Client(timeout=30) as client:
            r = client.get(url, params={"apiKey": api_key})
            r.raise_for_status()
            data = r.json()
        results = data.get("results") or []
        for bar in results:
            t = bar.get("t", 0)
            all_rows.append({
                "timestamp": pd.Timestamp(t, unit="ms", tz="America/New_York"),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v", 0),
            })
        next_url = data.get("next_url")
        url = f"{base}{next_url}" if next_url else None
        if url and "apiKey" not in url:
            url = f"{url}&apiKey={api_key}"
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def backfill_ticker_year(
    ticker: str,
    year: int,
    store_root: Path | None = None,
    api_key: str | None = None,
    use_massive_client: bool = False,
) -> Path | None:
    """Backfill one ticker for one year; write to store. Returns path if written. Set use_massive_client=True when massive is installed."""
    store_root = store_root or DEFAULT_STORE_ROOT
    from_ts = datetime(year, 1, 1)
    to_ts = datetime(year, 12, 31, 23, 59, 59)
    if use_massive_client:
        try:
            df = fetch_1m_bars_massive(ticker, from_ts, to_ts, api_key=api_key)
        except ImportError:
            df = fetch_1m_bars_polygon(ticker, from_ts, to_ts, api_key=api_key)
    else:
        df = fetch_1m_bars_polygon(ticker, from_ts, to_ts, api_key=api_key)
    if df.empty:
        return None
    store_root.mkdir(parents=True, exist_ok=True)
    ticker_dir = store_root / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_dir / f"{year}.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    from phinence.store.schemas import BAR_1M_SCHEMA
    table = pa.Table.from_pandas(df, schema=BAR_1M_SCHEMA, preserve_index=False)
    pq.write_table(table, path)
    return path


def backfill_universe_2y(
    tickers: list[str],
    store_root: Path | None = None,
    api_key: str | None = None,
    use_massive_client: bool = False,
) -> dict[str, list[Path]]:
    """Backfill 2+ years for each ticker. Returns ticker -> list of written paths."""
    from datetime import date
    today = date.today()
    written: dict[str, list[Path]] = {t: [] for t in tickers}
    for ticker in tickers:
        for y in range(today.year - 2, today.year + 1):
            p = backfill_ticker_year(
                ticker, y, store_root=store_root, api_key=api_key, use_massive_client=use_massive_client
            )
            if p:
                written[ticker].append(p)
    return written

"""
FINRA daily short sale volume. Phase 2C.

Uses public CDN: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
Format: pipe-delimited, Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market.
No API key required. Files posted by 6 PM ET on trade date.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

FINRA_CDN_CNMS = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
DEFAULT_SHORT_VOLUME_DIR = Path(__file__).resolve().parents[2] / "data" / "short_volume"


def fetch_finra_daily_short_volume(trade_date: date) -> pd.DataFrame:
    """
    Fetch consolidated NMS short volume for one trade date from FINRA CDN.
    Returns DataFrame with columns: date, symbol, short_volume, short_exempt_volume, total_volume, market.
    """
    datestr = trade_date.strftime("%Y%m%d")
    url = FINRA_CDN_CNMS.format(date=datestr)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        text = r.text
    lines = text.strip().split("\n")
    if not lines:
        return pd.DataFrame()
    # Header: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
    header = lines[0].split("|")
    rows = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) < 6:
            continue
        # Skip trailer row (record count)
        try:
            dt_str = parts[0].strip()
            symbol = parts[1].strip()
            if not symbol or not dt_str.isdigit():
                continue
            rows.append({
                "date": pd.Timestamp(dt_str) if len(dt_str) == 8 else trade_date,
                "symbol": symbol,
                "short_volume": int(parts[2].strip()) if parts[2].strip().isdigit() else 0,
                "short_exempt_volume": int(parts[3].strip()) if parts[3].strip().isdigit() else 0,
                "total_volume": int(parts[4].strip()) if parts[4].strip().isdigit() else 0,
                "market": parts[5].strip() if len(parts) > 5 else "",
            })
        except (ValueError, IndexError):
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def persist_short_volume(df: pd.DataFrame, trade_date: date, out_dir: Path | None = None) -> Path:
    """Write daily short volume to data/short_volume/YYYYMMDD.parquet."""
    out_dir = out_dir or DEFAULT_SHORT_VOLUME_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{trade_date.strftime('%Y%m%d')}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_short_volume(trade_date: date, data_dir: Path | None = None) -> pd.DataFrame:
    """Load persisted short volume for a date; empty DataFrame if missing."""
    data_dir = data_dir or DEFAULT_SHORT_VOLUME_DIR
    path = data_dir / f"{trade_date.strftime('%Y%m%d')}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def fetch_and_persist_daily(trade_date: date | None = None, out_dir: Path | None = None) -> Path | None:
    """Fetch from FINRA CDN for trade_date (default: latest trading day) and persist. Returns path or None on failure."""
    if trade_date is None:
        trade_date = date.today()
        # If weekend, use last Friday
        if trade_date.weekday() >= 5:
            trade_date -= timedelta(days=trade_date.weekday() - 4)
    df = fetch_finra_daily_short_volume(trade_date)
    if df.empty:
        return None
    return persist_short_volume(df, trade_date, out_dir=out_dir)

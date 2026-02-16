"""
Tradier: live 1m bars + options chain snapshots. Snapshot staleness per symbol.

Ported patterns from cablehead/python-tradier: quotes, expirations, chains(symbol, expiration).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx
import pandas as pd

TRADIER_SANDBOX = "https://sandbox.tradier.com/v1"
TRADIER_LIVE = "https://api.tradier.com/v1"

# Endpoints (cablehead-style)
ENDPOINTS = {"staging": TRADIER_SANDBOX + "/", "brokerage": TRADIER_LIVE + "/"}


def _headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or os.environ.get("TRADIER_ACCESS_TOKEN")
    if not key:
        raise ValueError("TRADIER_ACCESS_TOKEN required")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def get_quote(
    symbols: list[str] | str,
    base_url: str = TRADIER_SANDBOX,
    api_key: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any] | None:
    """
    Get quote(s). Ported from proshotv2/cablehead. Returns list of quote dicts
    or single quote; None if missing.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    url = f"{base_url.rstrip('/')}/markets/quotes"
    params = {"symbols": ",".join(s.upper() for s in symbols)}
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params, headers=_headers(api_key))
        r.raise_for_status()
        data = r.json()
    quotes = (data.get("quotes") or {}).get("quote")
    if quotes is None:
        return None
    return quotes if isinstance(quotes, list) else [quotes]


def fetch_live_1m_bars(
    ticker: str,
    session: str = "RTH",
    base_url: str = TRADIER_SANDBOX,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch today's 1m bars (live stream persisted to same schema as historical)."""
    url = f"{base_url}/markets/history"
    params = {"symbol": ticker.upper(), "interval": "1min", "session_filter": session}
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params, headers=_headers(api_key))
        r.raise_for_status()
        data = r.json()
    history = data.get("history") or {}
    days = history.get("day") or []
    rows = []
    for d in days:
        ts = d.get("time")
        if not ts:
            continue
        rows.append({
            "timestamp": pd.Timestamp(ts, tz="America/New_York"),
            "open": float(d.get("open", 0)),
            "high": float(d.get("high", 0)),
            "low": float(d.get("low", 0)),
            "close": float(d.get("close", 0)),
            "volume": int(d.get("volume", 0)),
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def options_expirations(
    symbol: str,
    base_url: str = TRADIER_SANDBOX,
    api_key: str | None = None,
) -> list[str]:
    """List option expiration dates (cablehead: markets/options/expirations)."""
    url = f"{base_url.rstrip('/')}/markets/options/expirations"
    params = {"symbol": symbol.upper()}
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params, headers=_headers(api_key))
        r.raise_for_status()
        data = r.json()
    exp = data.get("expirations") or {}
    dates = exp.get("date")
    return dates if isinstance(dates, list) else [dates] if dates else []


def fetch_chain_snapshot(
    ticker: str,
    base_url: str = TRADIER_SANDBOX,
    api_key: str | None = None,
    expiration: str | None = None,
) -> dict[str, Any]:
    """
    Options chain snapshot. If expiration is None, first expiration is used
    (cablehead: chains(symbol, expiration)). Staleness tracked by snapshot_ts.
    """
    if not expiration:
        exps = options_expirations(ticker, base_url=base_url, api_key=api_key)
        expiration = exps[0] if exps else None
    url = f"{base_url.rstrip('/')}/markets/options/chains"
    params = {"symbol": ticker.upper()}
    if expiration:
        params["expiration"] = expiration
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params, headers=_headers(api_key))
        r.raise_for_status()
        data = r.json()
    options = data.get("options") or {}
    return {
        "ticker": ticker.upper(),
        "snapshot_ts": datetime.utcnow().isoformat() + "Z",
        "options": options,
        "raw": data,
    }


def chain_snapshot_staleness_seconds(snapshot: dict[str, Any]) -> float:
    """Seconds since snapshot_ts. For V1 EOD use only; intraday staleness is high."""
    ts_str = (snapshot.get("snapshot_ts") or "").strip().rstrip("Z")
    if not ts_str:
        return float("inf")
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", ""))
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
    except Exception:
        return float("inf")
    return (datetime.utcnow() - dt).total_seconds()

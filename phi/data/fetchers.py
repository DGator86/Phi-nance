"""
phi.data.fetchers — OHLCV Data Fetchers
=========================================
Primary: yfinance (free, no API key).
Secondary: Alpha Vantage (requires AV_API_KEY env var).

All fetchers return a standardized DataFrame:
  Index: DatetimeIndex (tz-naive)
  Columns: open, high, low, close, volume
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Timeframe maps
# ─────────────────────────────────────────────────────────────────────────────

TIMEFRAMES = ["1D", "4H", "1H", "15m", "5m", "1m"]

_YF_INTERVAL: dict = {
    "1D":  "1d",
    "4H":  "1h",   # yfinance doesn't have 4H; aggregate below
    "1H":  "1h",
    "15m": "15m",
    "5m":  "5m",
    "1m":  "1m",
}

_YF_REQUIRES_SHORT_RANGE = {"15m", "5m", "1m"}   # yfinance only returns 60d for these


# ─────────────────────────────────────────────────────────────────────────────
# yfinance fetcher (primary)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(
    symbol: str,
    start,
    end,
    timeframe: str = "1D",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Yahoo Finance.

    Parameters
    ----------
    symbol    : ticker symbol (e.g. 'SPY')
    start     : start date (str or date)
    end       : end date (str or date)
    timeframe : one of TIMEFRAMES

    Returns
    -------
    Standardized DataFrame or None on failure.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    interval = _YF_INTERVAL.get(timeframe, "1d")
    start_str = str(start)[:10]
    end_str   = str(end)[:10]

    # yfinance intraday data limited to ~60 days for sub-hourly
    if timeframe in _YF_REQUIRES_SHORT_RANGE:
        end_dt   = pd.Timestamp(end_str).date()
        start_dt = pd.Timestamp(start_str).date()
        if (end_dt - start_dt).days > 59:
            start_str = str(end_dt - timedelta(days=59))

    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=True,
        actions=False,
    )

    if raw is None or raw.empty:
        return None

    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]

    # Ensure standard columns present
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = float("nan")

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)

    # Strip timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 4H: resample 1H → 4H
    if timeframe == "4H":
        df = (
            df.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["close"])
        )

    df = df.dropna(subset=["close"])
    return df if not df.empty else None


# ─────────────────────────────────────────────────────────────────────────────
# Alpha Vantage fetcher (secondary)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_alpha_vantage(
    symbol: str,
    start,
    end,
    timeframe: str = "1D",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Alpha Vantage (daily data only for free tier).
    Falls back to existing AlphaVantageFetcher if available.
    """
    try:
        from regime_engine.data_fetcher import AlphaVantageFetcher  # noqa: PLC0415
        av = AlphaVantageFetcher()
        if timeframe == "1D":
            raw = av.daily(symbol, outputsize="full")
        else:
            raw = av.intraday(symbol, outputsize="full")

        if raw is None or raw.empty:
            return None

        raw = raw[
            (raw.index >= pd.Timestamp(str(start)[:10])) &
            (raw.index <= pd.Timestamp(str(end)[:10]))
        ]
        raw.columns = [c.lower() for c in raw.columns]
        for col in ("open", "high", "low", "close", "volume"):
            if col not in raw.columns:
                raw[col] = float("nan")
        raw = raw[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        return raw if not raw.empty else None

    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Unified fetch interface
# ─────────────────────────────────────────────────────────────────────────────

VENDORS = ["yfinance", "alpha_vantage"]


def fetch(
    symbol: str,
    start,
    end,
    timeframe: str = "1D",
    vendor: str = "yfinance",
) -> Optional[pd.DataFrame]:
    """
    Unified fetch: route to the correct vendor fetcher.

    Returns standardized OHLCV DataFrame or None.
    """
    if vendor == "yfinance":
        return fetch_yfinance(symbol, start, end, timeframe)
    elif vendor == "alpha_vantage":
        df = fetch_alpha_vantage(symbol, start, end, timeframe)
        if df is None:
            return fetch_yfinance(symbol, start, end, timeframe)
        return df
    else:
        return fetch_yfinance(symbol, start, end, timeframe)


def dataset_summary(df: pd.DataFrame, symbol: str) -> dict:
    """Return a summary dict for display in the UI."""
    if df is None or df.empty:
        return {}
    close = df["close"].dropna()
    ret = (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0.0
    return {
        "symbol":     symbol.upper(),
        "rows":       len(df),
        "first_bar":  str(df.index[0])[:19],
        "last_bar":   str(df.index[-1])[:19],
        "open":       float(df["open"].iloc[0]),
        "last_close": float(close.iloc[-1]),
        "total_return": ret,
        "min_close":  float(close.min()),
        "max_close":  float(close.max()),
        "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else 0,
    }

"""
Convert Phi-nance bar store to DataFrame format expected by backtesting.py (kernc).

backtesting.py expects: DataFrame with columns Open, High, Low, Close, Volume and datetime index.
See: https://kernc.github.io/backtesting.py/
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def bar_store_to_bt_df(
    bar_store: Any,
    ticker: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    timeframe: str = "1D",
) -> pd.DataFrame:
    """
    Load 1m bars for ticker from bar_store and return a DataFrame suitable for backtesting.py.

    Columns: Open, High, Low, Close, Volume (capitalized). Index: datetime.
    If timeframe is '1D', 1m bars are resampled to daily OHLCV.

    Requires bar_store.read_1m_bars(ticker) returning DataFrame with timestamp, open, high, low, close, volume.
    """
    df = bar_store.read_1m_bars(ticker)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        return pd.DataFrame()

    if timeframe == "1D" or timeframe == "1d":
        resampled = df.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        resampled.columns = ["Open", "High", "Low", "Close", "Volume"]
    else:
        resampled = df[["open", "high", "low", "close", "volume"]].copy()
        resampled.columns = ["Open", "High", "Low", "Close", "Volume"]

    return resampled.astype(float)

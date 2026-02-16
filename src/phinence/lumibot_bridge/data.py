"""
Convert Phi-nance bar store (ParquetBarStore / InMemoryBarStore) to Lumibot pandas_data.

Lumibot PandasDataBacktesting expects:
  pandas_data: dict[Asset, Data] where Data(asset, df, timestep="minute"|"day")
  df: index name "datetime", dtype datetime64; columns open, high, low, close, volume (float)
  All datetimes timezone-aware (default America/New_York).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def _dataframe_to_lumibot_format(df: pd.DataFrame, timezone: str = "America/New_York") -> pd.DataFrame:
    """
    Convert Phi-nance 1m bar DataFrame (timestamp, open, high, low, close, volume)
    to Lumibot format: index datetime (tz-aware), columns open, high, low, close, volume (float).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "timestamp" not in out.columns:
        return pd.DataFrame()
    out["datetime"] = pd.to_datetime(out["timestamp"])
    out = out.set_index("datetime")[["open", "high", "low", "close", "volume"]].astype(float)
    out.index.name = "datetime"
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone, ambiguous="infer")
    return out.sort_index()


def bar_store_to_pandas_data(
    bar_store: Any,
    tickers: list[str],
    start: datetime | pd.Timestamp | str,
    end: datetime | pd.Timestamp | str,
    timestep: str = "minute",
    timezone: str = "America/New_York",
) -> tuple[dict[Any, Any], pd.Timestamp, pd.Timestamp]:
    """
    Build Lumibot pandas_data dict from a bar store that has read_1m_bars(ticker).

    Returns (pandas_data, datetime_start, datetime_end).
    pandas_data is dict[Asset, Data] for use with PandasDataBacktesting(pandas_data=..., datetime_start=..., datetime_end=...).
    Requires lumibot: pip install phi-nance[lumibot].
    """
    try:
        from lumibot.entities import Asset, Data
    except ImportError as e:
        raise ImportError("Lumibot is required for bar_store_to_pandas_data. Install with: pip install phi-nance[lumibot]") from e

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(timezone)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(timezone)

    pandas_data: dict[Any, Any] = {}
    for ticker in tickers:
        df = bar_store.read_1m_bars(ticker)
        if df is None or df.empty or len(df) < 10:
            continue
        df = _dataframe_to_lumibot_format(df, timezone=timezone)
        if df.empty:
            continue
        # Clip to requested range
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty or len(df) < 10:
            continue
        asset = Asset(symbol=ticker.upper(), asset_type=Asset.AssetType.STOCK)
        data = Data(asset, df, timestep=timestep)
        pandas_data[asset] = data

    if not pandas_data:
        raise ValueError("No bar data in store for the given tickers and date range")

    # Use first asset's Data datetime_start/datetime_end (set by Lumibot after trim)
    first_data = next(iter(pandas_data.values()))
    datetime_start = getattr(first_data, "datetime_start", start_ts)
    datetime_end = getattr(first_data, "datetime_end", end_ts)
    return pandas_data, pd.Timestamp(datetime_start), pd.Timestamp(datetime_end)

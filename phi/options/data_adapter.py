"""Helpers for adapting PostgreSQL options rows into backtesting-friendly frames."""

from __future__ import annotations

import pandas as pd

from phi.data.cache import fetch_and_cache


def adapt_for_backtesting(df: pd.DataFrame, timestamp_unit: str = "ms") -> pd.DataFrame:
    """Adapt raw options rows to a normalized dataframe for strategy research."""
    out = df.copy()
    if "quote_time" in out.columns:
        out["timestamp"] = pd.to_datetime(out["quote_time"], unit=timestamp_unit)
        out = out.set_index("timestamp").sort_index()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].sort_index()

    mapping = {
        "last": "close",
        "total_volume": "volume",
        "strike_price": "strike",
        "expiration_date": "expiration",
    }
    rename = {k: v for k, v in mapping.items() if k in out.columns}
    if rename:
        out = out.rename(columns=rename)

    for col, default in {"close": 0.0, "volume": 0.0}.items():
        if col not in out.columns:
            out[col] = default

    expected_option_fields = {
        "delta": pd.NA,
        "gamma": pd.NA,
        "theta": pd.NA,
        "vega": pd.NA,
        "rho": pd.NA,
        "open_interest": 0.0,
        "strike": pd.NA,
        "expiration": pd.NaT,
    }
    for col, default in expected_option_fields.items():
        if col not in out.columns:
            out[col] = default

    return out


def fetch_options_data(symbol: str, start: str, end: str, **kwargs) -> pd.DataFrame:
    """Fetch options rows via postgres vendor and adapt output for backtesting."""
    raw = fetch_and_cache(vendor="postgres", symbol=symbol, start=start, end=end, **kwargs)
    return adapt_for_backtesting(raw, timestamp_unit=kwargs.get("timestamp_unit", "ms"))

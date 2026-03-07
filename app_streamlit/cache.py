"""Cache helpers for Streamlit workbench data and indicator calculations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from phi.data import fetch_and_cache
from phi.indicators.simple import compute_indicator


@st.cache_data(ttl=3600, show_spinner=False)
def load_historical_data(symbol: str, start: str, end: str, timeframe: str, vendor: str) -> pd.DataFrame:
    """Fetch and cache OHLCV data for one symbol/config tuple."""
    return fetch_and_cache(vendor, symbol, timeframe, start, end)


@st.cache_data(ttl=1800, show_spinner=False)
def compute_indicator_signals(data: pd.DataFrame, indicator_name: str, params: dict[str, Any]) -> pd.Series:
    """Compute indicator series with cache keyed by name and params."""
    return compute_indicator(indicator_name, data, params)

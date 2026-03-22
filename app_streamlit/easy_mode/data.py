"""Cached OHLCV for easy mode with vendor fallback."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from phi.data import fetch_ohlcv_uw_then_yf
from phi.logging import get_logger

from app_streamlit.easy_mode.constants import LOOKBACK_DAYS

logger = get_logger(__name__)


@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv_cached(symbol: str, days: int, day_key: str) -> pd.DataFrame:
    """Load daily OHLCV; ``day_key`` busts cache once per calendar day."""
    sym = symbol.strip().upper()
    end = date.today()
    start = end - timedelta(days=max(days, 60))
    start_s, end_s = start.isoformat(), end.isoformat()
    try:
        df, _vendor = fetch_ohlcv_uw_then_yf(sym, start_s, end_s, timeframe="1D")
        return df
    except RuntimeError as exc:
        logger.warning("easy_mode OHLCV failed for %s: %s", sym, exc)
        raise RuntimeError(
            f"Could not load data for {sym}. Set UNUSUAL_WHALES_API_KEY or check network. ({exc})"
        ) from exc


def load_ohlcv(symbol: str, days: int | None = None) -> pd.DataFrame:
    d = days if days is not None else LOOKBACK_DAYS
    return load_ohlcv_cached(symbol, d, date.today().isoformat())

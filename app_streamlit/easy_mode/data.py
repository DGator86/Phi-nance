"""Cached OHLCV for easy mode with vendor fallback."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from phi.data import fetch_and_cache
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
    last_exc: Exception | None = None
    for vendor in ("unusual_whales", "yfinance"):
        try:
            df = fetch_and_cache(vendor, sym, "1D", start_s, end_s)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("easy_mode fetch %s via %s failed: %s", sym, vendor, exc)
    raise RuntimeError(f"Could not load data for {sym}. Set UNUSUAL_WHALES_API_KEY or check network. ({last_exc})")


def load_ohlcv(symbol: str, days: int | None = None) -> pd.DataFrame:
    d = days if days is not None else LOOKBACK_DAYS
    return load_ohlcv_cached(symbol, d, date.today().isoformat())

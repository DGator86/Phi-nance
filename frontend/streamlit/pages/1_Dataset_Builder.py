"""
Page 1 — Dataset Builder
========================

Fetch and cache OHLCV data from any supported vendor.
Stores result in st.session_state["ohlcv"] for downstream pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phinance.data import fetch_and_cache, list_cached_datasets
from phinance.config.settings import get_settings

st.set_page_config(page_title="Dataset Builder | Phi-nance", layout="wide")
st.title("1 · Dataset Builder")
st.caption("Fetch and cache OHLCV market data from any supported vendor.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Data Source")
    vendor = st.selectbox(
        "Vendor",
        ["alphavantage", "yfinance", "binance"],
        index=["alphavantage", "yfinance", "binance"].index(
            st.session_state.get("vendor", "alphavantage")
        ),
        help="yfinance: no API key needed. alphavantage: daily data via yfinance, intraday needs AV key.",
    )
    symbol = st.text_input("Symbol", value=st.session_state.get("symbol", "SPY"))
    timeframe = st.selectbox(
        "Timeframe",
        ["1D", "1H", "15m", "5m", "1m"],
        index=["1D", "1H", "15m", "5m", "1m"].index(
            st.session_state.get("timeframe", "1D")
        ),
    )
    start_date = st.date_input("Start Date", value=pd.Timestamp(st.session_state.get("start_date", "2022-01-01")))
    end_date   = st.date_input("End Date",   value=pd.Timestamp(st.session_state.get("end_date", "2024-12-31")))
    fetch_btn  = st.button("Fetch Data", type="primary", use_container_width=True)

# ── Cached datasets list ──────────────────────────────────────────────────────
with st.expander("📦 Cached Datasets", expanded=False):
    datasets = list_cached_datasets()
    if datasets:
        st.dataframe(
            pd.DataFrame(datasets)[["vendor", "symbol", "timeframe", "start", "end", "rows"]],
            use_container_width=True,
        )
    else:
        st.info("No cached datasets yet.")

# ── Fetch ─────────────────────────────────────────────────────────────────────
if fetch_btn:
    settings = get_settings()
    with st.spinner(f"Fetching {symbol} {timeframe} from {vendor}…"):
        try:
            df = fetch_and_cache(
                vendor    = vendor,
                symbol    = symbol,
                timeframe = timeframe,
                start     = str(start_date),
                end       = str(end_date),
                api_key   = settings.av_api_key or None,
            )
            st.session_state.update({
                "ohlcv":      df,
                "vendor":     vendor,
                "symbol":     symbol,
                "timeframe":  timeframe,
                "start_date": str(start_date),
                "end_date":   str(end_date),
            })
            st.success(f"Loaded **{len(df):,} bars** for {symbol} ({timeframe})")
        except Exception as exc:
            st.error(f"Fetch failed: {exc}")

# ── Preview ───────────────────────────────────────────────────────────────────
df = st.session_state.get("ohlcv")
if df is not None and not df.empty:
    st.subheader("Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Start", str(df.index[0])[:10])
    col3.metric("End", str(df.index[-1])[:10])
    st.dataframe(df.tail(20), use_container_width=True)

    # Candlestick chart
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
        )])
        fig.update_layout(
            title=f"{symbol} {timeframe}",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.line_chart(df["close"], height=300)
else:
    st.info("Fetch data using the sidebar controls to get started.")

st.divider()
st.markdown("**Next →** [2 · Indicator Selection](2_Indicator_Selection)")

"""
frontend.streamlit.components.data_preview
==========================================

Reusable data preview component: OHLCV table + candlestick chart.

Usage
-----
    from frontend.streamlit.components.data_preview import render_ohlcv_preview
    render_ohlcv_preview(df, symbol="SPY")
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_ohlcv_preview(
    df: pd.DataFrame,
    symbol: str = "",
    rows: int = 20,
    height: int = 350,
) -> None:
    """Render an OHLCV candlestick chart + data table.

    Parameters
    ----------
    df     : pd.DataFrame — OHLCV with DatetimeIndex
    symbol : str          — title label
    rows   : int          — number of tail rows to show in table
    height : int          — chart height in pixels
    """
    if df is None or df.empty:
        st.info("No data to preview.")
        return

    # Metrics strip
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Start", str(df.index[0])[:10])
    col3.metric("End", str(df.index[-1])[:10])
    col4.metric("Symbol", symbol or "—")

    # Candlestick
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )])
        fig.update_layout(
            title=f"{symbol} OHLCV" if symbol else "OHLCV",
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(df["close"], height=height)

    # Table
    with st.expander(f"Last {rows} bars"):
        st.dataframe(df.tail(rows), use_container_width=True)

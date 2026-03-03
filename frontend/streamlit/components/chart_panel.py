"""
frontend.streamlit.components.chart_panel
==========================================

Reusable Plotly chart components for the Phi-nance workbench.

Functions
---------
  equity_curve_chart(portfolio_value, index, initial_capital, title)
      — Returns a Plotly Figure with shaded equity curve + initial-cap line.
  candlestick_chart(ohlcv, title)
      — Candlestick / OHLCV chart.
  signal_overlay_chart(ohlcv, signals, title)
      — Close price overlaid with one or more indicator signal traces.
  drawdown_chart(portfolio_value, title)
      — Running drawdown expressed as a percentage.
  render_equity_curve(...)
      — Convenience wrapper that calls ``st.plotly_chart``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

# ── Optional plotly import ────────────────────────────────────────────────────

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False

# ── Colour palette ────────────────────────────────────────────────────────────

_COLORS = {
    "primary":    "#6366f1",
    "positive":   "#22c55e",
    "negative":   "#f43f5e",
    "neutral":    "#94a3b8",
    "surface":    "rgba(99,102,241,0.15)",
    "zero_line":  "#475569",
}

_LAYOUT_DEFAULTS: Dict[str, Any] = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(color="#eeeef2"),
    margin        = dict(l=4, r=4, t=36, b=4),
    legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ── Equity curve ─────────────────────────────────────────────────────────────


def equity_curve_chart(
    portfolio_value: Sequence[float],
    index: Optional[Any] = None,
    initial_capital: float = 100_000.0,
    title: str = "Equity Curve",
    height: int = 350,
) -> "go.Figure":
    """Build an equity-curve Plotly figure.

    Parameters
    ----------
    portfolio_value : sequence of floats — NAV per bar
    index           : date-like sequence aligned to ``portfolio_value``
    initial_capital : float — reference level drawn as horizontal line
    title           : str
    height          : int — figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for chart_panel. Install with: pip install plotly")

    x = list(index) if index is not None else list(range(len(portfolio_value)))
    pv = list(portfolio_value)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pv,
        name="Portfolio NAV",
        mode="lines",
        fill="tozeroy",
        fillcolor=_COLORS["surface"],
        line=dict(color=_COLORS["primary"], width=2),
    ))
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color=_COLORS["zero_line"],
        annotation_text=f"Initial ${initial_capital:,.0f}",
        annotation_position="bottom right",
        annotation_font_size=11,
    )
    fig.update_layout(
        title=title,
        height=height,
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── Candlestick ───────────────────────────────────────────────────────────────


def candlestick_chart(
    ohlcv: pd.DataFrame,
    title: str = "Price",
    height: int = 400,
) -> "go.Figure":
    """Build a candlestick chart from an OHLCV DataFrame.

    Parameters
    ----------
    ohlcv  : pd.DataFrame with ``open, high, low, close`` columns and DatetimeIndex
    title  : str
    height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    fig = go.Figure(data=[go.Candlestick(
        x=ohlcv.index,
        open=ohlcv.get("open",  ohlcv.get("Open")),
        high=ohlcv.get("high",  ohlcv.get("High")),
        low=ohlcv.get("low",   ohlcv.get("Low")),
        close=ohlcv.get("close", ohlcv.get("Close")),
        increasing_line_color=_COLORS["positive"],
        decreasing_line_color=_COLORS["negative"],
        name="OHLCV",
    )])
    fig.update_layout(
        title=title,
        height=height,
        xaxis_rangeslider_visible=False,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── Signal overlay ────────────────────────────────────────────────────────────

_SIGNAL_PALETTE = [
    "#a855f7", "#f59e0b", "#06b6d4", "#22c55e",
    "#f43f5e", "#6366f1", "#fb923c", "#e879f9",
]


def signal_overlay_chart(
    ohlcv: pd.DataFrame,
    signals: Dict[str, pd.Series],
    title: str = "Price + Signals",
    height: int = 380,
    scale_factor: float = 0.40,
) -> "go.Figure":
    """Overlay normalised indicator signals on the close price.

    Signals are rescaled to the price range and overlaid for visual inspection.

    Parameters
    ----------
    ohlcv         : pd.DataFrame
    signals       : dict ``{indicator_name: signal_series}``
    title         : str
    height        : int
    scale_factor  : float — how much of the price range to use for signals
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    close = ohlcv["close"]
    price_range = float(close.max() - close.min())
    mid = float(close.mean())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close,
        name="Close",
        line=dict(color=_COLORS["neutral"], width=1.5),
    ))

    for i, (name, sig) in enumerate(signals.items()):
        color = _SIGNAL_PALETTE[i % len(_SIGNAL_PALETTE)]
        overlay = mid + sig * price_range * scale_factor
        fig.add_trace(go.Scatter(
            x=sig.index, y=overlay,
            name=name,
            opacity=0.75,
            line=dict(color=color, width=1),
        ))

    fig.update_layout(title=title, height=height, **_LAYOUT_DEFAULTS)
    return fig


# ── Drawdown chart ────────────────────────────────────────────────────────────


def drawdown_chart(
    portfolio_value: Sequence[float],
    index: Optional[Any] = None,
    title: str = "Running Drawdown",
    height: int = 220,
) -> "go.Figure":
    """Build a running-drawdown chart.

    Parameters
    ----------
    portfolio_value : sequence of NAV floats
    index           : date-like sequence
    title           : str
    height          : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    import numpy as np

    pv = np.asarray(portfolio_value, dtype=float)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / np.where(peak > 0, peak, 1.0) * 100  # in percent

    x = list(index) if index is not None else list(range(len(dd)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dd,
        name="Drawdown %",
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(244,63,94,0.12)",
        line=dict(color=_COLORS["negative"], width=1.5),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=_COLORS["zero_line"])
    fig.update_layout(
        title=title,
        height=height,
        yaxis_ticksuffix="%",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ── Streamlit convenience wrappers ────────────────────────────────────────────


def render_equity_curve(
    portfolio_value: Sequence[float],
    index: Optional[Any] = None,
    initial_capital: float = 100_000.0,
    title: str = "Equity Curve",
    height: int = 350,
) -> None:
    """Render an equity curve directly into the Streamlit page.

    Requires ``streamlit`` to be importable.
    """
    import streamlit as st

    fig = equity_curve_chart(
        portfolio_value=portfolio_value,
        index=index,
        initial_capital=initial_capital,
        title=title,
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drawdown(
    portfolio_value: Sequence[float],
    index: Optional[Any] = None,
    title: str = "Running Drawdown",
    height: int = 220,
) -> None:
    """Render a drawdown chart directly into the Streamlit page."""
    import streamlit as st

    fig = drawdown_chart(
        portfolio_value=portfolio_value,
        index=index,
        title=title,
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True)

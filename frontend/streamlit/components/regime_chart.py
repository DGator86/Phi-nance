"""
frontend.streamlit.components.regime_chart
==========================================

Market-regime timeline visualisation for the Phi-nance workbench.

Functions
---------
  regime_timeline_chart(regime_series, title, height)
      — Colour-coded horizontal bar chart of regime phases over time.
  render_regime_timeline(...)
      — Convenience wrapper that calls ``st.plotly_chart``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False


# ── Colour mapping ────────────────────────────────────────────────────────────

_REGIME_COLORS: Dict[str, str] = {
    "TREND_UP":    "#22c55e",
    "TREND_DN":    "#f43f5e",
    "RANGE":       "#94a3b8",
    "BREAKOUT_UP": "#06b6d4",
    "BREAKOUT_DN": "#f97316",
    "HIGHVOL":     "#a855f7",
    "LOWVOL":      "#fbbf24",
    "UNKNOWN":     "#475569",
}

_LAYOUT_DEFAULTS: Dict[str, Any] = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(color="#eeeef2"),
    margin        = dict(l=4, r=4, t=36, b=4),
)


# ── Chart builders ────────────────────────────────────────────────────────────


def regime_timeline_chart(
    regime_series: pd.Series,
    title: str = "Market Regime Timeline",
    height: int = 120,
) -> "go.Figure":
    """Build a colour-coded regime timeline chart.

    Each bar segment represents a run of the same regime label.

    Parameters
    ----------
    regime_series : pd.Series of str — indexed by date/datetime
    title         : str
    height        : int — figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required for regime_chart")

    if regime_series is None or len(regime_series) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=height, **_LAYOUT_DEFAULTS)
        return fig

    # Group consecutive same-regime runs
    runs = []
    current_label = str(regime_series.iloc[0])
    current_start = regime_series.index[0]

    for idx, label in zip(regime_series.index[1:], regime_series.iloc[1:]):
        label = str(label)
        if label != current_label:
            runs.append((current_start, idx, current_label))
            current_label = label
            current_start = idx
    runs.append((current_start, regime_series.index[-1], current_label))

    fig = go.Figure()
    already_shown = set()

    for start, end, label in runs:
        color = _REGIME_COLORS.get(label, "#475569")
        show_legend = label not in already_shown
        already_shown.add(label)

        fig.add_trace(go.Bar(
            x=[(pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86_400],
            base=[pd.Timestamp(start)],
            y=["regime"],
            orientation="h",
            marker_color=color,
            marker_line_width=0,
            width=1.0,
            name=label,
            showlegend=show_legend,
            hovertemplate=f"{label}<br>{start} → {end}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        height=height,
        barmode="stack",
        xaxis=dict(type="date"),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        **_LAYOUT_DEFAULTS,
    )
    return fig


def regime_pie_chart(
    regime_series: pd.Series,
    title: str = "Regime Distribution",
    height: int = 280,
) -> "go.Figure":
    """Build a pie chart showing regime distribution.

    Parameters
    ----------
    regime_series : pd.Series of str
    title         : str
    height        : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not _HAS_PLOTLY:
        raise ImportError("plotly is required")

    counts = regime_series.value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [_REGIME_COLORS.get(l, "#475569") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.4,
        textinfo="percent+label",
    )])
    fig.update_layout(title=title, height=height, **_LAYOUT_DEFAULTS)
    return fig


# ── Streamlit convenience ─────────────────────────────────────────────────────


def render_regime_timeline(
    regime_series: pd.Series,
    title: str = "Market Regime Timeline",
    height: int = 120,
) -> None:
    """Render the regime timeline chart directly into the Streamlit page."""
    import streamlit as st

    fig = regime_timeline_chart(regime_series=regime_series, title=title, height=height)
    st.plotly_chart(fig, use_container_width=True)

"""
frontend.streamlit.components — Reusable Streamlit UI components.

Submodules
----------
  metric_cards  — KPI card row + verdict banner
  data_preview  — OHLCV preview table + basic stats
  run_selector  — Run-history dropdown selector
  chart_panel   — Equity curve, candlestick, signal overlay, drawdown charts
  regime_chart  — Regime timeline and pie charts
"""

from frontend.streamlit.components.metric_cards import (
    render_kpi_row,
    render_verdict_banner,
)
from frontend.streamlit.components.chart_panel import (
    equity_curve_chart,
    candlestick_chart,
    drawdown_chart,
    render_equity_curve,
    render_drawdown,
)
from frontend.streamlit.components.regime_chart import (
    regime_timeline_chart,
    regime_pie_chart,
    render_regime_timeline,
)

__all__ = [
    "render_kpi_row",
    "render_verdict_banner",
    "equity_curve_chart",
    "candlestick_chart",
    "drawdown_chart",
    "render_equity_curve",
    "render_drawdown",
    "regime_timeline_chart",
    "regime_pie_chart",
    "render_regime_timeline",
]

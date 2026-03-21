"""
Optional Dash monitor (demo) — SPY close chart on a 60s refresh.

Install: pip install -r requirements-dash.txt
Run from repo root: python app_dash/app.py

Use Streamlit easy/expert mode for regime + backtests; this is a lightweight second UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, callback, dcc, html
except ImportError as exc:
    raise SystemExit("Install Dash: pip install -r requirements-dash.txt") from exc

app = Dash(__name__)
app.title = "Phi-nance Dash"
app.layout = html.Div(
    style={"backgroundColor": "#0f172a", "color": "#e2e8f0", "padding": "1.5rem", "minHeight": "100vh"},
    children=[
        html.H2("Phi-nance Dash monitor (demo)"),
        html.P(
            "Auto-refresh SPY daily closes. Pair with Streamlit for full regime/backtest workflows.",
            style={"opacity": 0.85},
        ),
        dcc.Graph(id="chart"),
        dcc.Interval(id="tick", interval=60_000, n_intervals=0),
    ],
)


@callback(Output("chart", "figure"), Input("tick", "n_intervals"))
def _update_chart(_n: int) -> go.Figure:
    try:
        import pandas as pd
        import yfinance as yf

        raw = yf.download("SPY", period="1y", interval="1d", progress=False)
        if raw is None or raw.empty:
            return go.Figure()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]
        close = pd.Series(close).squeeze()
        fig = go.Figure(
            data=[go.Scatter(x=close.index, y=close.values, name="SPY", line=dict(color="#38bdf8"))]
        )
        fig.update_layout(template="plotly_dark", title="SPY — daily close", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")
        return fig
    except Exception:
        return go.Figure()


if __name__ == "__main__":
    app.run(debug=False, port=8050)

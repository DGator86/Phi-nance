#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Premium Dashboard — Production-Grade $250/mo SaaS
============================================================
Glassmorphism dark theme, animated Plotly charts, dynamic KPIs,
regime heatmaps, real-time feel, professional trading workstation.
"""

import os
import sys
import time
import random
import threading
import importlib
import hashlib
import warnings
from pathlib import Path

# Suppress Lumibot pandas FutureWarning (Series.__getitem__)
warnings.filterwarnings("ignore", category=FutureWarning, module="lumibot.entities.bars")
from datetime import date, datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Project root ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("IS_BACKTESTING", "True")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (BrokenPipeError, OSError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# THEME + CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
COLORS = {
    "purple": "#a855f7",
    "purple_dark": "#7c3aed",
    "purple_light": "#c084fc",
    "orange": "#f97316",
    "green": "#22c55e",
    "red": "#ef4444",
    "cyan": "#06b6d4",
    "yellow": "#eab308",
    "pink": "#ec4899",
    "bg_deep": "#08080c",
    "bg_card": "#16161e",
    "bg_elevated": "#1c1c27",
    "text": "#e8e8ed",
    "text_muted": "#8b8b9e",
    "text_dim": "#5a5a70",
}

CHART_COLORS = ["#a855f7", "#f97316", "#22c55e", "#06b6d4", "#eab308",
                "#ec4899", "#ef4444", "#c084fc"]

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,22,30,0.5)",
    font=dict(family="Inter, SF Pro Display, system-ui", color="#e8e8ed", size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor="rgba(168,85,247,0.06)",
        zerolinecolor="rgba(168,85,247,0.1)",
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor="rgba(168,85,247,0.06)",
        zerolinecolor="rgba(168,85,247,0.1)",
        showgrid=True,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(168,85,247,0.15)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1c1c27",
        bordercolor="#a855f7",
        font=dict(color="#e8e8ed", size=12),
    ),
)

REGIME_COLORS = {
    "TREND_UP": "#22c55e", "TREND_DN": "#ef4444",
    "RANGE": "#94a3b8", "BREAKOUT_UP": "#f97316",
    "BREAKOUT_DN": "#a855f7", "EXHAUST_REV": "#ec4899",
    "LOWVOL": "#06b6d4", "HIGHVOL": "#eab308",
}

REGIME_BINS = list(REGIME_COLORS.keys())

INDICATOR_CATALOG = {
    "RSI": {"desc": "Relative Strength Index", "icon": "📊",
            "params": {"rsi_period": (2, 50, 14), "oversold": (10, 50, 30), "overbought": (50, 95, 70)},
            "strategy": "strategies.rsi.RSIStrategy"},
    "MACD": {"desc": "Moving Avg Convergence Divergence", "icon": "📈",
             "params": {"fast_period": (2, 50, 12), "slow_period": (10, 100, 26), "signal_period": (2, 30, 9)},
             "strategy": "strategies.macd.MACDStrategy"},
    "Bollinger": {"desc": "Bollinger Bands Squeeze & Breakout", "icon": "🎯",
                  "params": {"bb_period": (5, 100, 20), "num_std": (1, 4, 2)},
                  "strategy": "strategies.bollinger.BollingerBands"},
    "Dual SMA": {"desc": "Golden Cross / Death Cross", "icon": "✕",
                 "params": {"fast_period": (2, 100, 10), "slow_period": (10, 300, 50)},
                 "strategy": "strategies.dual_sma.DualSMACrossover"},
    "Mean Reversion": {"desc": "Mean Reversion SMA Strategy", "icon": "↩",
                       "params": {"sma_period": (5, 200, 20)},
                       "strategy": "strategies.mean_reversion.MeanReversion"},
    "Breakout": {"desc": "Donchian Channel Breakout", "icon": "⚡",
                 "params": {"channel_period": (5, 100, 20)},
                 "strategy": "strategies.breakout.ChannelBreakout"},
    "Buy & Hold": {"desc": "Passive Long-Only Baseline", "icon": "🏦",
                   "params": {},
                   "strategy": "strategies.buy_and_hold.BuyAndHold"},
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def inject_css():
    """Inject the premium CSS stylesheet."""
    css_path = _ROOT / ".streamlit" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def phi_logo_sidebar():
    """Render premium logo in sidebar."""
    st.sidebar.markdown("""
    <div class="phi-logo-container">
        <div class="phi-logo-text">PHI-NANCE</div>
        <div class="phi-logo-sub">Quantitative Trading Intelligence</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(icon: str, title: str, badge: str = ""):
    """Render a styled section header."""
    badge_html = f'<span class="phi-section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="phi-section-header">
        <span class="phi-section-icon">{icon}</span>
        <span class="phi-section-title">{title}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def kpi_row(kpis: list):
    """Render a row of KPI cards. Each: (label, value, delta, delta_type)."""
    cards = ""
    for label, value, delta, delta_type in kpis:
        delta_class = {"positive": "positive", "negative": "negative"}.get(delta_type, "neutral")
        delta_html = f'<div class="phi-kpi-delta {delta_class}">{delta}</div>' if delta else ""
        cards += f"""
        <div class="phi-kpi-card">
            <div class="phi-kpi-label">{label}</div>
            <div class="phi-kpi-value">{value}</div>
            {delta_html}
        </div>"""
    st.markdown(f'<div class="phi-kpi-row">{cards}</div>', unsafe_allow_html=True)


def signal_badge(signal: str) -> str:
    """Return HTML for a signal badge."""
    cls = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}.get(signal.upper(), "hold")
    return f'<span class="phi-signal phi-signal-{cls}">{signal.upper()}</span>'


def status_dot(status: str) -> str:
    """Return HTML for a status indicator dot."""
    return f'<span class="phi-status-dot {status}"></span>'


def make_plotly_fig(**extra_layout) -> go.Figure:
    """Create a pre-themed Plotly figure."""
    fig = go.Figure()
    layout = {**PLOTLY_LAYOUT, **extra_layout}
    fig.update_layout(**layout)
    return fig


def format_currency(val, decimals=0):
    if val is None: return "—"
    if abs(val) >= 1e6: return f"${val/1e6:,.{decimals}f}M"
    if abs(val) >= 1e3: return f"${val/1e3:,.{decimals}f}K"
    return f"${val:,.{decimals}f}"


def format_pct(val, decimals=1):
    if val is None: return "—"
    return f"{val:+.{decimals}f}%"


def _load_strategy(module_cls: str):
    mod_path, cls_name = module_cls.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def _load_ohlcv_yf(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load OHLCV data via yfinance with caching."""
    cache_dir = _ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{symbol}_{start}_{end}".encode()).hexdigest()[:12]
    cache_path = cache_dir / f"{symbol}_{key}.parquet"

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    try:
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.columns = [c.lower() for c in df.columns]
            try:
                df.to_parquet(cache_path)
            except Exception:
                pass
            return df
    except Exception as e:
        st.error(f"Failed to fetch {symbol}: {e}")
    return None


def _extract_scalar(val):
    if isinstance(val, dict):
        for k in ("drawdown", "value", "max_drawdown", "return"):
            if k in val: return val[k]
        for v in val.values():
            if isinstance(v, (int, float)): return v
        return None
    return val


# ═══════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════
def build_candlestick_chart(df: pd.DataFrame, symbol: str = "SPY") -> go.Figure:
    """Premium candlestick chart with volume overlay."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        increasing_fillcolor=COLORS["green"],
        decreasing_fillcolor=COLORS["red"],
        name=symbol, line_width=1,
    ), row=1, col=1)

    # Volume bars
    vol_colors = [COLORS["green"] if c >= o else COLORS["red"]
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        marker_color=vol_colors, opacity=0.4,
        name="Volume", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=480,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{symbol} Price Action", font=dict(size=14, color="#c084fc")),
    )
    fig.update_xaxes(gridcolor="rgba(168,85,247,0.06)")
    fig.update_yaxes(gridcolor="rgba(168,85,247,0.06)")
    return fig


def build_equity_curve(portfolio_values: list, initial_capital: float = 100000) -> go.Figure:
    """Premium equity curve with gradient fill."""
    fig = make_plotly_fig(height=380)

    x = list(range(len(portfolio_values)))
    fig.add_trace(go.Scatter(
        x=x, y=portfolio_values,
        mode="lines",
        line=dict(color=COLORS["purple"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(168,85,247,0.08)",
        fillgradient=dict(type="vertical", colorscale=[
            [0, "rgba(168,85,247,0.0)"],
            [1, "rgba(168,85,247,0.12)"],
        ]),
        name="Portfolio Value",
        hovertemplate="<b>$%{y:,.0f}</b><extra></extra>",
    ))

    # Baseline
    fig.add_hline(
        y=initial_capital, line_dash="dot",
        line_color="rgba(148,163,184,0.3)",
        annotation_text=f"Start: ${initial_capital:,.0f}",
        annotation_font_color="#94a3b8",
    )

    fig.update_layout(
        title=dict(text="Portfolio Equity Curve", font=dict(size=14, color="#c084fc")),
        yaxis_title="Value ($)",
        xaxis_title="Trading Days",
    )
    return fig


def build_regime_heatmap(regime_probs: pd.DataFrame) -> go.Figure:
    """Dynamic regime probability heatmap."""
    # Sample last 120 bars for readability
    data = regime_probs.tail(120)
    fig = go.Figure(data=go.Heatmap(
        z=data.values.T,
        x=data.index.astype(str) if hasattr(data.index, 'astype') else list(range(len(data))),
        y=data.columns.tolist(),
        colorscale=[
            [0, "#08080c"],
            [0.2, "#1c1c27"],
            [0.4, "#7c3aed"],
            [0.7, "#a855f7"],
            [1.0, "#f97316"],
        ],
        showscale=True,
        colorbar=dict(title="Prob", tickformat=".0%"),
        hovertemplate="<b>%{y}</b><br>Bar: %{x}<br>Prob: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        title=dict(text="Regime Probability Field", font=dict(size=14, color="#c084fc")),
        xaxis_title="Time",
        yaxis_title="Regime",
        xaxis=dict(showticklabels=False, gridcolor="rgba(168,85,247,0.06)"),
    )
    return fig


def build_regime_donut(regime_probs: pd.DataFrame) -> go.Figure:
    """Donut chart of latest regime probabilities."""
    latest = regime_probs.iloc[-1].sort_values(ascending=False)
    colors = [REGIME_COLORS.get(r, "#94a3b8") for r in latest.index]

    fig = go.Figure(data=[go.Pie(
        labels=latest.index, values=latest.values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#08080c", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e8e8ed"),
        hovertemplate="<b>%{label}</b><br>%{value:.1%}<extra></extra>",
    )])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        showlegend=False,
        title=dict(text="Current Regime Mix", font=dict(size=13, color="#c084fc")),
        annotations=[dict(
            text=f"<b>{latest.index[0]}</b>",
            x=0.5, y=0.5, font_size=14, font_color=colors[0],
            showarrow=False,
        )],
    )
    return fig


def build_signals_chart(signals_df: pd.DataFrame) -> go.Figure:
    """Multi-line signal strength chart."""
    fig = make_plotly_fig(height=300)
    data = signals_df.tail(252)
    for i, col in enumerate(data.columns[:6]):
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col],
            mode="lines", name=col,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
            opacity=0.85,
        ))
    fig.update_layout(
        title=dict(text="Indicator Signal Strength", font=dict(size=14, color="#c084fc")),
        yaxis_title="Signal Z-Score",
    )
    return fig


def build_drawdown_chart(portfolio_values: list) -> go.Figure:
    """Underwater / drawdown chart."""
    pv = np.array(portfolio_values)
    running_max = np.maximum.accumulate(pv)
    dd = (pv - running_max) / running_max * 100

    fig = make_plotly_fig(height=250)
    fig.add_trace(go.Scatter(
        y=dd, mode="lines",
        line=dict(color=COLORS["red"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.08)",
        name="Drawdown",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Drawdown", font=dict(size=13, color="#c084fc")),
        yaxis_title="Drawdown %",
    )
    return fig


def build_returns_distribution(portfolio_values: list) -> go.Figure:
    """Returns distribution histogram."""
    pv = np.array(portfolio_values)
    returns = np.diff(pv) / pv[:-1] * 100

    fig = make_plotly_fig(height=250)
    fig.add_trace(go.Histogram(
        x=returns, nbinsx=50,
        marker_color=COLORS["purple"],
        opacity=0.7,
        name="Daily Returns",
        hovertemplate="%{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="rgba(148,163,184,0.4)")
    fig.update_layout(
        title=dict(text="Returns Distribution", font=dict(size=13, color="#c084fc")),
        xaxis_title="Daily Return %",
        yaxis_title="Frequency",
    )
    return fig


def build_composite_gauge(score: float, label: str = "Composite Signal") -> go.Figure:
    """Gauge chart for composite signal."""
    color = COLORS["green"] if score > 0.1 else COLORS["red"] if score < -0.1 else COLORS["yellow"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title=dict(text=label, font=dict(size=13, color="#c084fc")),
        number=dict(font=dict(size=28, color=color), valueformat="+.3f"),
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor="#5a5a70"),
            bar=dict(color=color),
            bgcolor="#16161e",
            borderwidth=0,
            steps=[
                dict(range=[-1, -0.3], color="rgba(239,68,68,0.12)"),
                dict(range=[-0.3, 0.3], color="rgba(234,179,8,0.08)"),
                dict(range=[0.3, 1], color="rgba(34,197,94,0.12)"),
            ],
            threshold=dict(line=dict(color="#e8e8ed", width=2), thickness=0.8, value=score),
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=220,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES HELPERS (scipy-based, used by Options Lab)
# ═══════════════════════════════════════════════════════════════════════════

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """Black-Scholes price for a vanilla call or put."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(0.0, S - K) if opt == "call" else max(0.0, K - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    except Exception:
        return 0.0


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> dict:
    """Black-Scholes first-order Greeks."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 1.0 if opt == "call" else -1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = float(norm.cdf(d1) if opt == "call" else -norm.cdf(-d1))
        gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
        # Theta: daily $ decay
        theta_raw = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if opt == "call":
            theta_raw -= r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta_raw += r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = float(theta_raw / 365)
        vega = float(S * norm.pdf(d1) * np.sqrt(T) / 100)  # per 1% IV
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}
    except Exception:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}


def _strategy_legs(strategy_type: str, spot: float, T: float, r: float, sigma: float):
    """
    Return list of (option_type, action, strike, qty) and net_debit for a strategy.
    action: +1 = long, -1 = short.  qty = number of options relative to 1-lot.
    """
    K = spot
    wing2 = spot * 0.02   # 2% wing
    wing5 = spot * 0.05   # 5% wing
    wing10 = spot * 0.10  # 10% wing

    if strategy_type == "long_call":
        legs = [("call", +1, K, 1)]
    elif strategy_type == "long_put":
        legs = [("put", +1, K, 1)]
    elif strategy_type == "bull_call_spread":
        legs = [("call", +1, K - wing2, 1), ("call", -1, K + wing2, 1)]
    elif strategy_type == "bear_put_spread":
        legs = [("put", +1, K + wing2, 1), ("put", -1, K - wing2, 1)]
    elif strategy_type == "iron_condor":
        legs = [
            ("call", -1, K + wing5,  1),
            ("call", +1, K + wing10, 1),
            ("put",  -1, K - wing5,  1),
            ("put",  +1, K - wing10, 1),
        ]
    elif strategy_type == "straddle":
        legs = [("call", +1, K, 1), ("put", +1, K, 1)]
    elif strategy_type == "covered_call":
        # Model as short OTM call (stock leg excluded from option P&L)
        legs = [("call", -1, K + wing5, 1)]
    elif strategy_type == "cash_secured_put":
        legs = [("put", -1, K - wing5, 1)]
    else:
        legs = [("call", +1, K, 1)]

    # Net debit (positive = cost, negative = credit received)
    net = sum(action * _bs_price(spot, strike, T, r, sigma, ot)
              for ot, action, strike, _ in legs)
    return legs, net


def _strategy_payoff_at_expiry(strategy_type: str, S_range: np.ndarray, spot: float,
                                T: float, r: float, sigma: float) -> tuple:
    """Return (pnl_array, label) for payoff-at-expiry diagram."""
    legs, net_debit = _strategy_legs(strategy_type, spot, T, r, sigma)

    pnl = np.zeros_like(S_range, dtype=float)
    for opt_type, action, strike, qty in legs:
        if opt_type == "call":
            intrinsic = np.maximum(S_range - strike, 0.0)
        else:
            intrinsic = np.maximum(strike - S_range, 0.0)
        pnl += action * qty * intrinsic

    pnl -= net_debit  # subtract what we paid (add back credit received)

    # For covered_call: add underlying P&L (bought 1 share at spot)
    if strategy_type == "covered_call":
        pnl += (S_range - spot)

    label = f"Net {'debit' if net_debit > 0 else 'credit'}: ${abs(net_debit):.2f}/share"
    return pnl, label


def build_payoff_diagram(strategy_type: str, spot: float, sigma: float,
                         dte: int, r: float = 0.05) -> go.Figure:
    """Interactive payoff-at-expiry diagram for any supported strategy."""
    T = max(dte, 1) / 365.0
    S_range = np.linspace(spot * 0.70, spot * 1.30, 300)
    pnl, leg_label = _strategy_payoff_at_expiry(strategy_type, S_range, spot, T, r, sigma)

    # Current P&L (T = T, marked-to-market)
    current_pnl = np.zeros_like(S_range, dtype=float)
    legs, net_debit = _strategy_legs(strategy_type, spot, T, r, sigma)
    for opt_type, action, strike, qty in legs:
        for i, s in enumerate(S_range):
            T_now = T * 0.5  # halfway through
            current_pnl[i] += action * qty * _bs_price(s, strike, T_now, r, sigma, opt_type)
    current_pnl -= net_debit
    if strategy_type == "covered_call":
        current_pnl += S_range - spot

    fig = make_plotly_fig(height=400)

    # Shaded profit / loss zones at expiry
    profit_mask = pnl >= 0
    fig.add_trace(go.Scatter(
        x=np.concatenate([S_range[profit_mask], S_range[profit_mask][::-1]]),
        y=np.concatenate([pnl[profit_mask], np.zeros(profit_mask.sum())]),
        fill="toself", fillcolor="rgba(34,197,94,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    loss_mask = pnl < 0
    if loss_mask.any():
        fig.add_trace(go.Scatter(
            x=np.concatenate([S_range[loss_mask], S_range[loss_mask][::-1]]),
            y=np.concatenate([pnl[loss_mask], np.zeros(loss_mask.sum())]),
            fill="toself", fillcolor="rgba(239,68,68,0.07)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

    # P&L at current time (mark-to-market)
    fig.add_trace(go.Scatter(
        x=S_range, y=current_pnl, mode="lines", name="Current P&L (50% DTE)",
        line=dict(color=COLORS["cyan"], width=1.5, dash="dot"),
        hovertemplate="Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra>50% DTE</extra>",
    ))

    # P&L at expiry
    fig.add_trace(go.Scatter(
        x=S_range, y=pnl, mode="lines", name="P&L at Expiry",
        line=dict(color=COLORS["orange"], width=2.5),
        hovertemplate="Spot: $%{x:.2f}<br>P&L: $%{y:.2f}<extra>Expiry</extra>",
    ))

    fig.add_hline(y=0, line_color="rgba(148,163,184,0.35)", line_dash="dot")
    fig.add_vline(x=spot, line_color=COLORS["purple"], line_dash="dash",
                  annotation_text=f"Spot: ${spot:.2f}",
                  annotation_font_color=COLORS["purple_light"],
                  annotation_position="top right")

    strat_label = strategy_type.replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"Payoff Diagram — {strat_label} &nbsp;|&nbsp; {leg_label}",
            font=dict(size=13, color=COLORS["purple_light"]),
        ),
        xaxis_title="Underlying Price at Expiry ($)",
        yaxis_title="P&L per Share ($)",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MFT ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
def _load_base_config():
    import yaml
    cfg_path = _ROOT / "regime_engine" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_mft_pipeline(ohlcv: pd.DataFrame, cfg: dict = None) -> Optional[dict]:
    """Run the full MFT pipeline and return all outputs."""
    try:
        from regime_engine.scanner import RegimeEngine
        if cfg is None:
            cfg = _load_base_config()
        engine = RegimeEngine(cfg)
        return engine.run(ohlcv)
    except Exception as e:
        st.error(f"MFT Pipeline Error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: COMMAND CENTER (Home)
# ═══════════════════════════════════════════════════════════════════════════
def page_command_center():
    """Main dashboard / command center page."""

    # ── Market ticker banner ─────────────────────────────────────────────
    tickers_data = st.session_state.get("ticker_data", {})
    if tickers_data:
        items = ""
        for sym, data in list(tickers_data.items())[:8]:
            price = data.get("price", 0)
            chg = data.get("change_pct", 0)
            cls = "up" if chg >= 0 else "down"
            arrow = "▲" if chg >= 0 else "▼"
            items += f"""
            <div class="phi-ticker-item">
                <span class="phi-ticker-symbol">{sym}</span>
                <span class="phi-ticker-price">${price:,.2f}</span>
                <span class="phi-ticker-change {cls}">{arrow} {chg:+.2f}%</span>
            </div>"""
        st.markdown(f'<div class="phi-ticker-banner">{items}</div>', unsafe_allow_html=True)

    # ── Hero section ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem;">
        <div style="font-size: 0.75rem; color: #8b8b9e; text-transform: uppercase;
                    letter-spacing: 0.2em; margin-bottom: 0.5rem;">
            Market Field Theory Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats from session
    config = st.session_state.get("workbench_config", {})
    last_results = st.session_state.get("last_results", {})
    mft_out = st.session_state.get("mft_output", {})

    tr = last_results.get("total_return")
    sharpe = last_results.get("sharpe")
    dd = last_results.get("max_drawdown")
    accuracy = last_results.get("accuracy")
    cap = config.get("initial_capital", 100_000)

    kpi_row([
        ("Portfolio Value", format_currency(cap), None, "neutral"),
        ("Total Return", format_pct(tr * 100 if tr else None), "Latest Run", "positive" if tr and tr > 0 else "negative"),
        ("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "—", None, "neutral"),
        ("Max Drawdown", format_pct(dd * 100 if dd else None), None, "negative" if dd else "neutral"),
        ("Accuracy", format_pct(accuracy * 100 if accuracy else None), None, "positive" if accuracy and accuracy > 0.5 else "neutral"),
    ])

    st.markdown("")

    # ── Quick Actions ────────────────────────────────────────────────────
    section_header("⚡", "Quick Actions", "COMMAND CENTER")
    cols = st.columns(5)
    with cols[0]:
        if st.button("🔍 Scan Market", use_container_width=True, key="cmd_scan"):
            st.session_state["active_page"] = "Scanner"
            st.rerun()
    with cols[1]:
        if st.button("🧪 Run Backtest", use_container_width=True, key="cmd_bt"):
            st.session_state["active_page"] = "Workbench"
            st.rerun()
    with cols[2]:
        if st.button("📊 Options Lab", use_container_width=True, key="cmd_opts"):
            st.session_state["active_page"] = "Options Lab"
            st.rerun()
    with cols[3]:
        if st.button("🔬 MFT Analysis", use_container_width=True, key="cmd_mft"):
            st.session_state["active_page"] = "MFT Blender"
            st.rerun()
    with cols[4]:
        if st.button("🤖 AI Advisor", use_container_width=True, key="cmd_ai"):
            st.session_state["active_page"] = "AI Advisor"
            st.rerun()

    # ── Charts section ───────────────────────────────────────────────────
    st.markdown("")
    section_header("📊", "Market Overview")

    sym = config.get("symbols", ["SPY"])[0] if config.get("symbols") else "SPY"
    overview_sym = st.text_input("Symbol", value=sym, key="overview_sym",
                                  label_visibility="collapsed",
                                  placeholder="Enter symbol (e.g. SPY)")

    @st.cache_data(ttl=3600, show_spinner=False)
    def get_overview_data(symbol, days=365):
        end = date.today()
        start = end - timedelta(days=days)
        return _load_ohlcv_yf(symbol, str(start), str(end))

    with st.spinner("Loading market data..."):
        df = get_overview_data(overview_sym)

    if df is not None and not df.empty:
        # Store ticker data
        price = float(df["close"].iloc[-1])
        prev_price = float(df["close"].iloc[-2]) if len(df) > 1 else price
        chg_pct = (price - prev_price) / prev_price * 100
        st.session_state.setdefault("ticker_data", {})[overview_sym] = {
            "price": price, "change_pct": chg_pct
        }

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = build_candlestick_chart(df.tail(120), overview_sym)
            st.plotly_chart(fig, use_container_width=True, key="overview_candle")
        with col2:
            # Performance metrics
            ret_1m = (df["close"].iloc[-1] / df["close"].iloc[-22] - 1) * 100 if len(df) > 22 else 0
            ret_3m = (df["close"].iloc[-1] / df["close"].iloc[-66] - 1) * 100 if len(df) > 66 else 0
            ret_ytd = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            vol_20 = df["close"].pct_change().tail(20).std() * np.sqrt(252) * 100

            st.markdown(f"""
            <div class="phi-glass-card" style="margin-bottom: 1rem;">
                <div class="phi-section-title" style="margin-bottom: 12px;">{overview_sym} Stats</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                    <div>
                        <div class="phi-kpi-label">Last Price</div>
                        <div class="phi-kpi-value" style="font-size:1.2rem;">${price:,.2f}</div>
                    </div>
                    <div>
                        <div class="phi-kpi-label">Daily Change</div>
                        <div class="phi-kpi-delta {'positive' if chg_pct >= 0 else 'negative'}"
                             style="font-size:1.1rem;">{'▲' if chg_pct >= 0 else '▼'} {chg_pct:+.2f}%</div>
                    </div>
                    <div>
                        <div class="phi-kpi-label">1M Return</div>
                        <div class="phi-kpi-delta {'positive' if ret_1m >= 0 else 'negative'}">{ret_1m:+.1f}%</div>
                    </div>
                    <div>
                        <div class="phi-kpi-label">3M Return</div>
                        <div class="phi-kpi-delta {'positive' if ret_3m >= 0 else 'negative'}">{ret_3m:+.1f}%</div>
                    </div>
                    <div>
                        <div class="phi-kpi-label">YTD Return</div>
                        <div class="phi-kpi-delta {'positive' if ret_ytd >= 0 else 'negative'}">{ret_ytd:+.1f}%</div>
                    </div>
                    <div>
                        <div class="phi-kpi-label">20D Volatility</div>
                        <div style="color: #f97316; font-weight:600;">{vol_20:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Run MFT if data available
        if st.button("⚙ Run MFT Analysis", key="overview_mft", use_container_width=True):
            with st.spinner("Running Market Field Theory pipeline..."):
                mft_out = _run_mft_pipeline(df)
                if mft_out:
                    st.session_state["mft_output"] = mft_out
                    st.session_state["mft_symbol"] = overview_sym

        if st.session_state.get("mft_output") and st.session_state.get("mft_symbol") == overview_sym:
            mft = st.session_state["mft_output"]
            col_a, col_b = st.columns(2)
            with col_a:
                if "regime_probs" in mft:
                    fig = build_regime_heatmap(mft["regime_probs"])
                    st.plotly_chart(fig, use_container_width=True, key="overview_heatmap")
            with col_b:
                if "regime_probs" in mft:
                    fig = build_regime_donut(mft["regime_probs"])
                    st.plotly_chart(fig, use_container_width=True, key="overview_donut")

            if "mix" in mft:
                mix = mft["mix"]
                score = float(mix["composite_signal"].iloc[-1]) if "composite_signal" in mix.columns else 0
                c_field = float(mix["c_field"].iloc[-1]) if "c_field" in mix.columns else 0
                c_con = float(mix["c_consensus"].iloc[-1]) if "c_consensus" in mix.columns else 0
                c_liq = float(mix["c_liquidity"].iloc[-1]) if "c_liquidity" in mix.columns else 0
                overall = c_field * c_con * c_liq

                g1, g2, g3 = st.columns(3)
                with g1:
                    fig = build_composite_gauge(score, "Composite Signal")
                    st.plotly_chart(fig, use_container_width=True, key="gauge_comp")
                with g2:
                    fig = build_composite_gauge(overall, "Overall Confidence")
                    st.plotly_chart(fig, use_container_width=True, key="gauge_conf")
                with g3:
                    signal = "BUY" if score > 0.15 and overall > 0.1 else "SELL" if score < -0.15 and overall > 0.1 else "HOLD"
                    st.markdown(f"""
                    <div class="phi-glass-card" style="text-align:center; padding:2rem;">
                        <div class="phi-kpi-label" style="margin-bottom:12px;">MFT Signal</div>
                        <div style="font-size:2rem; margin-bottom:8px;">
                            {signal_badge(signal)}
                        </div>
                        <div style="color:#8b8b9e; font-size:0.8rem; margin-top:12px;">
                            Score: {score:+.3f}<br>
                            Confidence: {overall:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: WORKBENCH (Backtesting)
# ═══════════════════════════════════════════════════════════════════════════
def page_workbench():
    """Premium backtesting workbench."""
    section_header("🧪", "Live Backtest Workbench", "PRO")

    # ── Step 1: Dataset ──────────────────────────────────────────────────
    with st.expander("Step 1 — Dataset Configuration", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            wb_sym = st.text_input("Symbol(s)", value="SPY", key="wb_sym",
                                    help="Comma-separated: SPY, QQQ, AAPL")
        with c2:
            wb_start = st.date_input("Start Date", value=date(2020, 1, 1), key="wb_start")
        with c3:
            wb_end = st.date_input("End Date", value=date(2024, 12, 31), key="wb_end")

        c4, c5 = st.columns(2)
        with c4:
            wb_capital = st.number_input("Initial Capital ($)", value=100_000,
                                          min_value=1_000, step=10_000, key="wb_capital")
        with c5:
            wb_tf = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m"], key="wb_tf")

        if st.button("📥 Fetch & Cache Data", type="primary", use_container_width=True, key="wb_fetch"):
            symbols = [s.strip().upper() for s in wb_sym.split(",") if s.strip()]
            dfs = {}
            progress = st.progress(0, text="Fetching data...")
            for i, sym in enumerate(symbols):
                df = _load_ohlcv_yf(sym, str(wb_start), str(wb_end))
                if df is not None and not df.empty:
                    dfs[sym] = df
                progress.progress((i + 1) / len(symbols), text=f"Fetched {sym}...")
            progress.empty()

            if dfs:
                st.session_state["wb_dataset"] = dfs
                st.session_state["workbench_config"] = {
                    "symbols": list(dfs.keys()),
                    "start": datetime.combine(wb_start, datetime.min.time()),
                    "end": datetime.combine(wb_end, datetime.min.time()),
                    "initial_capital": float(wb_capital),
                    "timeframe": wb_tf,
                    "benchmark": list(dfs.keys())[0],
                }
                bars = sum(len(d) for d in dfs.values())
                st.success(f"Dataset ready: {', '.join(dfs.keys())} — {bars:,} bars cached")

        # Show data preview
        if st.session_state.get("wb_dataset"):
            dfs = st.session_state["wb_dataset"]
            primary_sym = list(dfs.keys())[0]
            df = dfs[primary_sym]
            fig = build_candlestick_chart(df.tail(120), primary_sym)
            st.plotly_chart(fig, use_container_width=True, key="wb_candle")

    # ── Step 2: Indicator Selection ──────────────────────────────────────
    with st.expander("Step 2 — Indicator Selection", expanded=True):
        selected_indicators = {}
        cols = st.columns(4)
        for i, (name, info) in enumerate(INDICATOR_CATALOG.items()):
            with cols[i % 4]:
                enabled = st.checkbox(f"{info['icon']} **{name}**", key=f"wb_ind_{name}")
                if enabled:
                    st.caption(info["desc"])
                    params = {}
                    for pname, (lo, hi, default) in info["params"].items():
                        params[pname] = st.slider(
                            pname, lo, hi, default,
                            key=f"wb_p_{name}_{pname}"
                        )
                    selected_indicators[name] = {"enabled": True, "params": params}

    # ── Step 3: Blending ─────────────────────────────────────────────────
    blend_method = "weighted_sum"
    blend_weights = {}
    if len(selected_indicators) >= 2:
        with st.expander("Step 3 — Signal Blending", expanded=True):
            blend_method = st.selectbox("Blend Mode",
                ["weighted_sum", "regime_weighted", "voting", "phiai_chooses"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="wb_blend")
            for name in selected_indicators:
                blend_weights[name] = st.slider(
                    f"Weight: {name}", 0.0, 1.0,
                    round(1.0 / len(selected_indicators), 2), 0.05,
                    key=f"wb_wt_{name}")

    # ── Step 4: Run ──────────────────────────────────────────────────────
    st.markdown("---")
    config = st.session_state.get("workbench_config")

    if not config or not selected_indicators:
        st.info("Complete Steps 1-2 to run a backtest.")
        return

    if st.button("🚀 Run Backtest", type="primary", use_container_width=True, key="wb_run"):
        _execute_backtest(config, selected_indicators, blend_method, blend_weights)


def _execute_backtest(config, indicators, blend_method, blend_weights):
    """Execute backtest with premium result display."""
    progress = st.progress(0, text="Initializing backtest...")

    try:
        # If single indicator, load that strategy directly
        if len(indicators) == 1:
            name = list(indicators.keys())[0]
            info = INDICATOR_CATALOG[name]
            strat_cls = _load_strategy(info["strategy"])
            p_defaults = {k: default for k, (_, _, default) in info["params"].items()}
            p_user = indicators[name].get("params", {})
            params = {**p_defaults, **p_user}
            params["symbol"] = config["symbols"][0]
        else:
            try:
                strat_cls = _load_strategy(
                    "strategies.blended_workbench_strategy.BlendedWorkbenchStrategy")
                params = {
                    "symbol": config["symbols"][0],
                    "indicators": indicators,
                    "blend_method": blend_method,
                    "blend_weights": blend_weights,
                    "signal_threshold": 0.15,
                    "lookback_bars": 200,
                }
            except Exception:
                name = list(indicators.keys())[0]
                info = INDICATOR_CATALOG[name]
                strat_cls = _load_strategy(info["strategy"])
                params = {"symbol": config["symbols"][0]}

        # Run in thread
        result_holder = [None]
        exc_holder = [None]

        def _av_datasource():
            from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
            return AlphaVantageFixedDataSource

        def run_bt():
            try:
                os.environ["IS_BACKTESTING"] = "True"
                av_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")
                results, strat = strat_cls.run_backtest(
                    datasource_class=_av_datasource(),
                    backtesting_start=config["start"],
                    backtesting_end=config["end"],
                    budget=config["initial_capital"],
                    benchmark_asset=config.get("benchmark", "SPY"),
                    parameters=params, api_key=av_key,
                    show_plot=False, show_tearsheet=False,
                    save_tearsheet=False, show_indicators=False,
                    show_progress_bar=False, quiet_logs=True,
                )
                result_holder[0] = (results, strat)
            except Exception as e:
                exc_holder[0] = e

        th = threading.Thread(target=run_bt)
        th.start()
        pct = 5
        start_t = time.time()
        while th.is_alive():
            time.sleep(0.4)
            elapsed = time.time() - start_t
            pct = min(95, 5 + int(elapsed * 1.5))
            progress.progress(pct / 100, text=f"Running backtest... {pct}%")

        if exc_holder[0]:
            progress.empty()
            st.error(f"Backtest failed: {exc_holder[0]}")
            return

        progress.progress(1.0, text="Complete!")
        time.sleep(0.3)
        progress.empty()

        if result_holder[0] is None:
            st.error("No results returned.")
            return

        results, strat = result_holder[0]
        _display_premium_results(results, strat, config)

    except Exception as e:
        progress.empty()
        st.error(str(e))
        st.exception(e)


def _display_premium_results(results, strat, config):
    """Premium result display with animated charts."""
    cap = config.get("initial_capital", 100_000)

    tr = _extract_scalar(getattr(results, "total_return", None)
                         or (results.get("total_return") if isinstance(results, dict) else None))
    cagr = _extract_scalar(getattr(results, "cagr", None)
                           or (results.get("cagr") if isinstance(results, dict) else None))
    dd = _extract_scalar(getattr(results, "max_drawdown", None)
                         or (results.get("max_drawdown") if isinstance(results, dict) else None))
    sharpe = _extract_scalar(getattr(results, "sharpe", None)
                             or (results.get("sharpe") if isinstance(results, dict) else None))

    pv = getattr(results, "portfolio_value", None)
    if pv is None and isinstance(results, dict):
        pv = results.get("portfolio_value", [])
    if pv is None:
        pv = []

    end_cap = pv[-1] if pv and len(pv) else cap
    net_pl = end_cap - cap
    net_pct = (net_pl / cap) * 100 if cap else 0

    # Try accuracy
    acc = None
    try:
        if strat and hasattr(strat, "prediction_log"):
            from strategies.prediction_tracker import compute_prediction_accuracy
            sc = compute_prediction_accuracy(strat)
            acc = sc.get("accuracy")
    except Exception:
        pass

    # Store for command center
    st.session_state["last_results"] = {
        "total_return": tr, "cagr": cagr, "max_drawdown": dd,
        "sharpe": sharpe, "accuracy": acc,
    }

    st.markdown("---")
    section_header("📈", "Backtest Results", "COMPLETE")

    # KPI Row
    kpi_row([
        ("Start Capital", format_currency(cap), None, "neutral"),
        ("End Capital", format_currency(end_cap), f"{net_pct:+.1f}%",
         "positive" if net_pl >= 0 else "negative"),
        ("Net P/L", f"${net_pl:+,.0f}", None, "positive" if net_pl >= 0 else "negative"),
        ("CAGR", format_pct(cagr * 100 if cagr else None), None,
         "positive" if cagr and cagr > 0 else "negative"),
        ("Sharpe", f"{sharpe:.2f}" if sharpe else "—", None, "neutral"),
        ("Max DD", format_pct(dd * 100 if dd else None), None, "negative" if dd else "neutral"),
    ])

    # Charts
    if pv and len(pv) > 2:
        tab1, tab2, tab3, tab4 = st.tabs(["Equity Curve", "Drawdown", "Distribution", "Trade Log"])
        with tab1:
            fig = build_equity_curve(pv, cap)
            st.plotly_chart(fig, use_container_width=True, key="bt_equity")
        with tab2:
            fig = build_drawdown_chart(pv)
            st.plotly_chart(fig, use_container_width=True, key="bt_dd")
        with tab3:
            fig = build_returns_distribution(pv)
            st.plotly_chart(fig, use_container_width=True, key="bt_dist")
        with tab4:
            if strat and hasattr(strat, "prediction_log") and strat.prediction_log:
                log_df = pd.DataFrame(strat.prediction_log)
                st.dataframe(log_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trade log available for this strategy.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MFT BLENDER
# ═══════════════════════════════════════════════════════════════════════════
def page_mft_blender():
    """MFT Blender with full pipeline visualization."""
    section_header("🔬", "MFT Blender — Full Pipeline Analysis", "ADVANCED")

    st.caption("All parameters map directly to the MFT config. "
               "Changes propagate through Taxonomy → Gates → MSL → Mixer → Projections.")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        mft_sym = st.text_input("Symbol", value="SPY", key="mft_sym")
    with c2:
        mft_start = st.date_input("From", value=date(2020, 1, 1), key="mft_start")
    with c3:
        mft_end = st.date_input("To", value=date(2024, 12, 31), key="mft_end")

    if st.button("🚀 Run Full MFT Pipeline", type="primary", use_container_width=True, key="mft_run"):
        with st.spinner("Loading data..."):
            ohlcv = _load_ohlcv_yf(mft_sym, str(mft_start), str(mft_end))
        if ohlcv is None or ohlcv.empty:
            st.error("No data available.")
            return

        with st.spinner("Running Market Field Theory pipeline..."):
            out = _run_mft_pipeline(ohlcv)

        if out is None:
            return

        st.session_state["mft_output"] = out
        st.session_state["mft_symbol"] = mft_sym
        st.session_state["mft_ohlcv"] = ohlcv

    out = st.session_state.get("mft_output")
    if out is None:
        st.info("Configure parameters and run the pipeline to see results.")
        return

    sym = st.session_state.get("mft_symbol", "SPY")
    ohlcv = st.session_state.get("mft_ohlcv")

    # ── Stage 1: Features ────────────────────────────────────────────────
    with st.expander("STAGE 1 — Feature Engine", expanded=False):
        feat_df = out.get("features")
        if feat_df is not None:
            cols = list(feat_df.columns)
            sel = st.multiselect("Features to display", cols, default=cols[:6], key="mft_feat_sel")
            if sel:
                fig = make_plotly_fig(height=300)
                data = feat_df[sel].tail(252)
                for i, c in enumerate(sel):
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data[c], name=c, mode="lines",
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
                    ))
                fig.update_layout(title=dict(text="Computed Features", font=dict(size=13, color="#c084fc")))
                st.plotly_chart(fig, use_container_width=True, key="mft_feat_chart")
            st.caption(f"{len(cols)} features computed")

    # ── Stage 2: Taxonomy Logits ─────────────────────────────────────────
    with st.expander("STAGE 2 — Taxonomy Logits (Kingdom → Genus)", expanded=True):
        logits_df = out.get("logits")
        if logits_df is not None:
            groups = {
                "Kingdom (DIR, NDR, TRN)": [c for c in logits_df.columns if c in ("DIR", "NDR", "TRN")],
                "Phylum (LV, NV, HV)": [c for c in logits_df.columns if c in ("LV", "NV", "HV")],
            }
            for gname, gcols in groups.items():
                if gcols:
                    fig = make_plotly_fig(height=220)
                    data = logits_df[gcols].tail(252)
                    for i, c in enumerate(gcols):
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data[c], name=c, mode="lines",
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                        ))
                    fig.update_layout(title=dict(text=gname, font=dict(size=12, color="#c084fc")))
                    st.plotly_chart(fig, use_container_width=True, key=f"mft_logit_{gname[:3]}")

    # ── Stage 3: Regime Probabilities ────────────────────────────────────
    with st.expander("STAGE 3 — Probability Field", expanded=True):
        regime_df = out.get("regime_probs")
        if regime_df is not None:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                fig = build_regime_heatmap(regime_df)
                st.plotly_chart(fig, use_container_width=True, key="mft_regime_heat")
            with col_b:
                fig = build_regime_donut(regime_df)
                st.plotly_chart(fig, use_container_width=True, key="mft_regime_donut")

            # Latest bar metrics
            latest = regime_df.iloc[-1].sort_values(ascending=False)
            regime_cols = st.columns(4)
            for i, (r, p) in enumerate(latest.head(8).items()):
                with regime_cols[i % 4]:
                    color = REGIME_COLORS.get(r, "#94a3b8")
                    st.markdown(f"""
                    <div style="text-align:center; padding:8px; border:1px solid {color}33;
                                border-radius:8px; background:{color}0a;">
                        <div style="color:{color}; font-weight:700; font-size:0.85rem;">{r}</div>
                        <div style="color:#e8e8ed; font-size:1.1rem; font-weight:600;">{p:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Stage 4: Signals & Weights ───────────────────────────────────────
    with st.expander("STAGE 4 — Indicator Signals & Weights", expanded=False):
        signals_df = out.get("signals")
        weights_df = out.get("weights")
        if signals_df is not None:
            tab_s, tab_w = st.tabs(["Signals", "Validity Weights"])
            with tab_s:
                fig = build_signals_chart(signals_df)
                st.plotly_chart(fig, use_container_width=True, key="mft_signals")
            with tab_w:
                if weights_df is not None:
                    fig = make_plotly_fig(height=280)
                    data = weights_df.tail(252)
                    for i, c in enumerate(data.columns[:6]):
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data[c], name=c, mode="lines",
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
                        ))
                    fig.update_layout(title=dict(text="Validity Weights", font=dict(size=13, color="#c084fc")))
                    st.plotly_chart(fig, use_container_width=True, key="mft_weights")

    # ── Stage 5: Mixer / Composite ───────────────────────────────────────
    with st.expander("STAGE 5 — Mixer / Composite Score", expanded=True):
        mix_df = out.get("mix")
        if mix_df is not None:
            conf_cols = [c for c in mix_df.columns
                         if c in ("composite_signal", "score", "c_field", "c_consensus", "c_liquidity")]
            if conf_cols:
                fig = make_plotly_fig(height=300)
                data = mix_df[conf_cols].tail(252)
                for i, c in enumerate(conf_cols):
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data[c], name=c, mode="lines",
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                    ))
                fig.update_layout(title=dict(text="Composite & Confidence Metrics", font=dict(size=13, color="#c084fc")))
                st.plotly_chart(fig, use_container_width=True, key="mft_mixer")

            # Gauges
            last = mix_df.iloc[-1]
            score = float(last.get("composite_signal", 0))
            c_f = float(last.get("c_field", 0))
            c_c = float(last.get("c_consensus", 0))
            c_l = float(last.get("c_liquidity", 0))
            overall = c_f * c_c * c_l

            g1, g2, g3 = st.columns(3)
            with g1:
                fig = build_composite_gauge(score, "Composite Signal")
                st.plotly_chart(fig, use_container_width=True, key="mft_g_comp")
            with g2:
                fig = build_composite_gauge(overall, "Overall Confidence")
                st.plotly_chart(fig, use_container_width=True, key="mft_g_conf")
            with g3:
                signal = "BUY" if score > 0.15 and overall > 0.1 else \
                         "SELL" if score < -0.15 and overall > 0.1 else "HOLD"
                if overall >= 0.1:
                    st.success(f"Signal is tradeable — confidence {overall:.3f}")
                else:
                    st.warning(f"Below trade threshold — confidence {overall:.3f}")
                st.markdown(f"<div style='text-align:center; margin-top:1rem;'>{signal_badge(signal)}</div>",
                           unsafe_allow_html=True)

    # ── Stage 6: Projections ─────────────────────────────────────────────
    with st.expander("STAGE 6 — Projection Engine AR(1)", expanded=False):
        projections = out.get("projections", {})
        proj_exp = projections.get("expected")
        proj_var = projections.get("variance")
        if proj_exp is not None:
            tab_e, tab_v = st.tabs(["Expected Value", "Variance"])
            with tab_e:
                fig = make_plotly_fig(height=280)
                data = proj_exp.tail(252)
                for i, c in enumerate(data.columns[:6]):
                    fig.add_trace(go.Scatter(
                        x=data.index, y=data[c], name=c, mode="lines",
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
                    ))
                fig.update_layout(title=dict(text="Expected Next Value (AR(1))", font=dict(size=13, color="#c084fc")))
                st.plotly_chart(fig, use_container_width=True, key="mft_proj_exp")
            with tab_v:
                if proj_var is not None:
                    fig = make_plotly_fig(height=280)
                    data = proj_var.tail(252)
                    for i, c in enumerate(data.columns[:6]):
                        fig.add_trace(go.Scatter(
                            x=data.index, y=data[c], name=c, mode="lines",
                            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
                        ))
                    fig.update_layout(title=dict(text="Mixture Variance (Uncertainty)", font=dict(size=13, color="#c084fc")))
                    st.plotly_chart(fig, use_container_width=True, key="mft_proj_var")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SCANNER
# ═══════════════════════════════════════════════════════════════════════════
def page_scanner():
    """Universe regime scanner."""
    section_header("🔍", "Universe Regime Scanner", "LIVE")

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        universe_raw = st.text_input("Universe (comma-separated)",
            value="SPY, QQQ, AAPL, NVDA, TSLA, MSFT, AMZN, GLD",
            key="scan_universe")
    with c2:
        scan_start = st.date_input("From", value=date(2022, 1, 1), key="scan_start")
    with c3:
        scan_end = st.date_input("To", value=date(2024, 12, 31), key="scan_end")

    if st.button("🔍 Scan Universe", type="primary", use_container_width=True, key="scan_go"):
        tickers = [t.strip().upper() for t in universe_raw.split(",") if t.strip()]
        progress = st.progress(0, text="Scanning...")

        universe_data = {}
        for i, sym in enumerate(tickers):
            df = _load_ohlcv_yf(sym, str(scan_start), str(scan_end))
            if df is not None and not df.empty:
                universe_data[sym] = df
            progress.progress((i + 1) / len(tickers), text=f"Fetching {sym}...")

        if not universe_data:
            progress.empty()
            st.error("No data fetched.")
            return

        # Try MFT scanner
        try:
            from regime_engine.scanner import UniverseScanner
            import yaml
            cfg_path = _ROOT / "regime_engine" / "config.yaml"
            scanner = UniverseScanner(config_path=str(cfg_path))
            progress.progress(0.9, text="Running MFT analysis...")
            scan_df = scanner.scan(universe_data, sort_by="score")
            st.session_state["scan_results"] = scan_df
            st.session_state["scan_ts"] = datetime.now().strftime("%H:%M:%S")
            progress.progress(1.0, text="Scan complete!")
            time.sleep(0.3)
            progress.empty()
        except Exception as e:
            progress.empty()
            st.warning(f"MFT scanner unavailable: {e}")
            # Fallback: simple performance table
            rows = []
            for sym, df in universe_data.items():
                ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
                vol = df["close"].pct_change().std() * np.sqrt(252) * 100
                rows.append({"ticker": sym, "return_%": round(ret, 2), "volatility_%": round(vol, 2),
                             "last_price": round(float(df["close"].iloc[-1]), 2)})
            st.session_state["scan_results"] = pd.DataFrame(rows)

    scan_df = st.session_state.get("scan_results")
    if scan_df is not None and not scan_df.empty:
        ts = st.session_state.get("scan_ts", "")
        if ts:
            st.caption(f"Last scan: {ts}")

        # Display results table
        st.dataframe(scan_df, use_container_width=True, hide_index=True)

        # Regime probability bars
        prob_cols = [c for c in scan_df.columns if c.startswith("p_")]
        if prob_cols and "ticker" in scan_df.columns:
            section_header("📊", "Regime Probabilities by Ticker")
            prob_data = scan_df.set_index("ticker")[prob_cols].rename(columns=lambda c: c.replace("p_", ""))

            fig = go.Figure()
            for i, regime in enumerate(prob_data.columns):
                color = REGIME_COLORS.get(regime, CHART_COLORS[i % len(CHART_COLORS)])
                fig.add_trace(go.Bar(
                    name=regime, x=prob_data.index, y=prob_data[regime],
                    marker_color=color, opacity=0.85,
                ))
            fig.update_layout(**PLOTLY_LAYOUT, height=400, barmode="stack",
                             title=dict(text="Regime Probability Stack", font=dict(size=14, color="#c084fc")))
            st.plotly_chart(fig, use_container_width=True, key="scan_regime_bar")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: AI ADVISOR
# ═══════════════════════════════════════════════════════════════════════════
def page_ai_advisor():
    """AI-powered trading advisor (Plutus + Ollama)."""
    section_header("🤖", "AI Trading Advisor", "PLUTUS")

    st.caption("Powered by [0xroyce/plutus](https://ollama.com/0xroyce/plutus) — "
               "LLaMA 3.1-8B fine-tuned on 394 finance & trading books. "
               "Runs locally via Ollama.")

    with st.expander("Connection Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ollama_host = st.text_input("Ollama Host", value="http://localhost:11434", key="ai_host")
        with c2:
            model = st.text_input("Model", value="0xroyce/plutus", key="ai_model")

    st.markdown("---")
    tab_ask, tab_bt, tab_journal = st.tabs(["Ask Plutus", "Backtest", "Trade Journal"])

    with tab_ask:
        section_header("💬", "Market Consultation")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            ask_sym = st.text_input("Symbol", value="SPY", key="ai_ask_sym")
        with c2:
            ask_start = st.date_input("From", value=date(2023, 1, 1), key="ai_ask_start")
        with c3:
            ask_end = st.date_input("To", value=date(2024, 12, 31), key="ai_ask_end")

        ask_conf = st.slider("Minimum confidence threshold", 0.50, 0.95, 0.60, 0.05, key="ai_conf")

        if st.button("🧠 Get Recommendation", type="primary", use_container_width=True, key="ai_ask"):
            with st.spinner("Consulting Plutus AI..."):
                try:
                    ohlcv = _load_ohlcv_yf(ask_sym, str(ask_start), str(ask_end))
                    if ohlcv is None or ohlcv.empty:
                        st.error("No data available.")
                        return

                    mft_out = None
                    try:
                        mft_out = _run_mft_pipeline(ohlcv)
                    except Exception:
                        pass

                    from regime_engine.plutus_advisor import PlutusAdvisor, build_market_brief
                    advisor = PlutusAdvisor(model=model, host=ollama_host, min_conf=ask_conf)

                    if not advisor.is_available():
                        st.error("Plutus model not available. Ensure Ollama is running and the model is pulled.")
                        return

                    ohlcv_sum, mft_sig = build_market_brief(ohlcv, mft_out)
                    price = float(ohlcv["close"].iloc[-1])
                    decision = advisor.recommend(ask_sym, ohlcv_sum, mft_sig, price)
                    st.session_state["plutus_decision"] = decision
                except Exception as e:
                    st.error(f"Error: {e}")

        dec = st.session_state.get("plutus_decision")
        if dec:
            st.markdown("---")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown(f"""
                <div class="phi-glass-card" style="text-align:center;">
                    <div class="phi-kpi-label">Signal</div>
                    <div style="margin:12px 0;">{signal_badge(dec.signal)}</div>
                </div>
                """, unsafe_allow_html=True)
            with d2:
                st.metric("Confidence", f"{dec.confidence:.0%}")
            with d3:
                st.metric("Actionable", "YES" if dec.is_actionable(ask_conf) else "NO")
            st.markdown(f"**Reasoning:** {dec.reasoning}")
            st.markdown(f"**Risk Note:** {dec.risk_note}")

    with tab_bt:
        st.info("Configure Ollama connection and run Plutus-guided backtests. "
                "Each bar consults the LLM — expect slower execution.")

    with tab_journal:
        st.info("Trade journal appears after running Ask Plutus or a backtest.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
def page_ml_models():
    """ML model status and training."""
    section_header("🧠", "ML Model Hub", "MODELS")

    ML_MODELS = [
        ("Random Forest", "models/classifier_rf.pkl", "sklearn"),
        ("Gradient Boosting", "models/classifier_gb.pkl", "sklearn"),
        ("Logistic Regression", "models/classifier_lr.pkl", "sklearn"),
        ("LightGBM", "models/classifier_lgb.txt", "lgb"),
    ]

    cols = st.columns(len(ML_MODELS))
    for col, (name, path, kind) in zip(cols, ML_MODELS):
        full_path = _ROOT / path
        exists = full_path.exists()
        with col:
            status_color = COLORS["green"] if exists else COLORS["red"]
            status_icon = "●" if exists else "○"
            size = f"{full_path.stat().st_size / 1024:.1f} KB" if exists else "Missing"
            st.markdown(f"""
            <div class="phi-glass-card" style="text-align:center; min-height:140px;">
                <div style="color:{status_color}; font-size:1.2rem; margin-bottom:8px;">{status_icon}</div>
                <div style="font-weight:700; color:#e8e8ed; font-size:0.9rem;">{name}</div>
                <div style="color:#8b8b9e; font-size:0.75rem; margin-top:4px;">{size}</div>
                <div style="color:#5a5a70; font-size:0.7rem; margin-top:2px;">{path}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Training data status
    csv_path = _ROOT / "historical_regime_features.csv"
    if csv_path.exists():
        try:
            n_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            st.success(f"Training data ready: {n_rows:,} rows — full MFT features")
        except Exception:
            st.warning("CSV exists but could not be read.")
    else:
        st.warning("No training CSV found. Generate using the Fetch Data flow.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 Generate Training Data", use_container_width=True, key="ml_gen",
                      disabled=csv_path.exists()):
            import subprocess
            with st.status("Generating MFT training data...", expanded=True) as s:
                r = subprocess.run([sys.executable, "-X", "utf8", str(_ROOT / "generate_training_data.py")],
                                   capture_output=True, text=True, encoding="utf-8")
                st.code(r.stdout + r.stderr)
                s.update(label="Done" if r.returncode == 0 else "Failed",
                         state="complete" if r.returncode == 0 else "error")
    with c2:
        if st.button("🧠 Train All Models", type="primary", use_container_width=True,
                      key="ml_train", disabled=not csv_path.exists()):
            import subprocess
            with st.status("Training models...", expanded=True) as s:
                r = subprocess.run([sys.executable, "-X", "utf8", str(_ROOT / "train_ml_classifier.py")],
                                   capture_output=True, text=True, encoding="utf-8")
                st.code(r.stdout + r.stderr)
                s.update(label="Done" if r.returncode == 0 else "Failed",
                         state="complete" if r.returncode == 0 else "error")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
def page_system():
    """System status and health checks."""
    section_header("⚙", "System Status", "HEALTH")

    if st.button("🔍 Run Health Check", type="primary", use_container_width=True, key="sys_check"):
        try:
            from engine_health import run_engine_health_check
            health = run_engine_health_check()
            st.session_state["health"] = health
        except Exception as e:
            st.session_state["health"] = {"ok": False, "error": str(e), "components": {}, "optional": {}}

    health = st.session_state.get("health")
    if health:
        if health.get("ok"):
            st.success("All MFT pipeline components operational.")
        else:
            st.warning("One or more components have issues.")

        if health.get("components"):
            cols = st.columns(3)
            for i, (name, info) in enumerate(health["components"].items()):
                with cols[i % 3]:
                    ok = info.get("ok", False)
                    color = COLORS["green"] if ok else COLORS["red"]
                    st.markdown(f"""
                    <div class="phi-glass-card" style="min-height:80px;">
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="color:{color}; font-size:1.2rem;">{'●' if ok else '○'}</span>
                            <span style="font-weight:700; color:#e8e8ed;">{name}</span>
                        </div>
                        <div style="color:#8b8b9e; font-size:0.8rem; margin-top:6px;">
                            {info.get('message', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # Run History
    st.markdown("---")
    section_header("📋", "Run History")
    try:
        from phi.run_config import RunHistory
        hist = RunHistory()
        runs = hist.list_runs()
        if runs:
            for r in runs[:10]:
                with st.expander(f"Run: {r['run_id']}"):
                    st.json(r.get("results", {}))
        else:
            st.info("No runs recorded yet.")
    except Exception:
        st.info("Run history module not available.")

    # Cache Manager
    st.markdown("---")
    section_header("💾", "Cache Manager")
    try:
        from phi.data import list_cached_datasets
        datasets = list_cached_datasets()
        if datasets:
            st.dataframe(pd.DataFrame(datasets), use_container_width=True, hide_index=True)
        else:
            st.info("No cached datasets.")
    except Exception:
        st.info("Cache manager not available.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OPTIONS LAB
# ═══════════════════════════════════════════════════════════════════════════

_OPTIONS_STRATEGIES = {
    "long_call":        ("Long Call",          "🟢", "Buy a call → profit from upside moves beyond strike"),
    "long_put":         ("Long Put",           "🔴", "Buy a put → profit from downside moves below strike"),
    "bull_call_spread": ("Bull Call Spread",   "📈", "Buy lower call / sell higher call — defined-risk bull"),
    "bear_put_spread":  ("Bear Put Spread",    "📉", "Buy higher put / sell lower put — defined-risk bear"),
    "iron_condor":      ("Iron Condor",        "🦅", "Sell OTM strangle, buy outer wings — range-bound income"),
    "straddle":         ("Long Straddle",      "⚡", "Buy ATM call + put — profit from large moves either way"),
    "covered_call":     ("Covered Call",       "🏦", "Long shares + sell OTM call — income on existing position"),
    "cash_secured_put": ("Cash-Secured Put",   "💰", "Sell OTM put secured by cash — acquire stock at a discount"),
}


def _run_options_sim(
    ohlcv: pd.DataFrame,
    strategy_type: str,
    initial_capital: float,
    position_pct: float,
    iv_assumption: float,
    dte_days: int,
    exit_profit_pct: float,
    exit_stop_pct: float,
    r: float = 0.05,
) -> dict:
    """
    Black-Scholes mark-to-market simulation for any multi-leg strategy.

    Entry: open of bar, notional = capital × position_pct.
    Exit : when cum_return >= profit target OR <= -stop loss OR DTE elapsed.
    P&L  : sum of BS mid-price changes across all legs × notional.
    """
    close = ohlcv["close"].values
    n = len(close)
    if n < 10:
        return {
            "portfolio_value": [initial_capital], "total_return": 0,
            "cagr": 0, "max_drawdown": 0, "sharpe": 0, "trades": [],
        }

    capital = float(initial_capital)
    pv_series = [capital]
    trades = []
    in_pos = False
    entry_day = 0
    entry_spot = 0.0
    entry_premium = 0.0  # net $ cost of position per-share
    notional = 0.0

    def _position_value(spot: float, days_remaining: int) -> float:
        """Current value per-share of the full multi-leg structure."""
        T_now = max(days_remaining, 0) / 365.0
        legs, _ = _strategy_legs(strategy_type, entry_spot, max(dte_days, 1) / 365.0, r, iv_assumption)
        val = 0.0
        for opt_type, action, strike, qty in legs:
            val += action * qty * _bs_price(spot, strike, T_now, r, iv_assumption, opt_type)
        if strategy_type == "covered_call":
            val += spot - entry_spot   # underlying P&L
        return val

    for i in range(1, n):
        spot = float(close[i])

        if not in_pos:
            # Enter at start of bar
            T_entry = max(dte_days, 1) / 365.0
            _, net_debit = _strategy_legs(strategy_type, spot, T_entry, r, iv_assumption)
            entry_premium = net_debit
            entry_spot = spot
            entry_day = i
            notional = capital * position_pct
            in_pos = True
            pv_series.append(capital)
            continue

        days_remaining = max(dte_days - (i - entry_day), 0)
        current_val = _position_value(spot, days_remaining)
        position_pnl = current_val - entry_premium

        # Dollar P&L scaled to notional
        scale = notional / max(abs(entry_premium), 1e-6) if entry_premium != 0 else 0.0
        dollar_pnl = position_pnl * scale
        cur_capital = capital + dollar_pnl
        pv_series.append(cur_capital)

        # Exit conditions
        cum_ret = position_pnl / max(abs(entry_premium), 1e-6) if entry_premium != 0 else 0.0
        expired = days_remaining <= 0
        take_profit = cum_ret >= exit_profit_pct
        stop_loss = cum_ret <= -exit_stop_pct

        if take_profit or stop_loss or expired:
            trades.append({
                "entry_bar": entry_day, "exit_bar": i,
                "entry_spot": round(entry_spot, 2), "exit_spot": round(spot, 2),
                "pnl_$": round(dollar_pnl, 2),
                "cum_ret_%": round(cum_ret * 100, 2),
                "exit_reason": "profit" if take_profit else ("stop" if stop_loss else "expiry"),
                "days_held": i - entry_day,
            })
            capital = cur_capital
            in_pos = False

    pv_arr = np.array(pv_series)
    total_return = float(pv_arr[-1] / initial_capital - 1) if initial_capital else 0
    n_years = max(len(ohlcv) / 252, 1)
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else total_return
    peak = np.maximum.accumulate(pv_arr)
    dd = (pv_arr - peak) / np.where(peak > 0, peak, 1e-8)
    max_dd = float(np.min(dd))
    pv_ret = np.diff(pv_arr) / np.maximum(pv_arr[:-1], 1e-8)
    sharpe = float(np.mean(pv_ret) / np.std(pv_ret) * np.sqrt(252)) if np.std(pv_ret) > 1e-10 else 0.0

    return {
        "portfolio_value": list(pv_arr),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": trades,
    }


def _execute_options_backtest_page(
    symbol, start, end, capital, strategy_type,
    delta, iv, dte, position_pct, exit_profit, exit_stop,
):
    """Fetch data, run sim, store results in session state."""
    with st.spinner(f"Fetching {symbol} OHLCV data..."):
        ohlcv = _load_ohlcv_yf(symbol, start, end)
    if ohlcv is None or ohlcv.empty:
        st.error(f"No OHLCV data available for **{symbol}**. Check the symbol or date range.")
        return

    strat_label = _OPTIONS_STRATEGIES[strategy_type][0]
    with st.spinner(f"Running {strat_label} backtest on {symbol}…"):
        if strategy_type in ("long_call", "long_put"):
            # Use the validated phi engine for vanilla strategies
            from phi.options.backtest import run_options_backtest
            results = run_options_backtest(
                ohlcv=ohlcv, symbol=symbol,
                strategy_type=strategy_type,
                initial_capital=capital, position_pct=position_pct,
                delta_assumption=delta,
                exit_profit_pct=exit_profit, exit_stop_pct=-exit_stop,
            )
        else:
            results = _run_options_sim(
                ohlcv=ohlcv, strategy_type=strategy_type,
                initial_capital=capital, position_pct=position_pct,
                iv_assumption=iv, dte_days=dte,
                exit_profit_pct=exit_profit, exit_stop_pct=exit_stop,
            )

    st.session_state["opt_results"] = results
    st.session_state["opt_ohlcv"] = ohlcv
    st.session_state["opt_capital_used"] = capital
    st.session_state["opt_strategy_used"] = strategy_type
    st.session_state["opt_symbol_used"] = symbol
    st.rerun()


def _build_iv_surface_fig(spot: float, sigma_base: float, r: float = 0.05) -> go.Figure:
    """3-D implied-volatility surface (skew × term-structure mock)."""
    strikes = np.linspace(spot * 0.75, spot * 1.25, 20)
    tenors = [7, 14, 30, 60, 90, 180]

    # Mimic typical skew: OTM puts have higher IV, term-structure upward
    Z = []
    for t in tenors:
        row = []
        for k in strikes:
            moneyness = np.log(k / spot)
            skew = -0.4 * moneyness + 0.1 * moneyness ** 2   # put skew
            term_adj = 1 + 0.002 * (t - 30)
            iv_here = np.clip(sigma_base * (1 + skew) * term_adj, 0.05, 2.0)
            row.append(iv_here * 100)
        Z.append(row)

    fig = go.Figure(data=[go.Surface(
        x=strikes, y=tenors, z=Z,
        colorscale=[
            [0.0, "#08080c"], [0.2, "#7c3aed"],
            [0.5, "#a855f7"], [0.8, "#f97316"],
            [1.0, "#fbbf24"],
        ],
        opacity=0.90,
        colorbar=dict(title="IV %", tickformat=".0f", len=0.6),
        hovertemplate="Strike: $%{x:.0f}<br>DTE: %{y}d<br>IV: %{z:.1f}%<extra></extra>",
    )])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        scene=dict(
            xaxis=dict(title="Strike ($)", gridcolor="rgba(168,85,247,0.1)"),
            yaxis=dict(title="DTE (days)", gridcolor="rgba(168,85,247,0.1)"),
            zaxis=dict(title="IV (%)", gridcolor="rgba(168,85,247,0.1)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        title=dict(text="Implied Volatility Surface (Illustrative Skew)", font=dict(size=13, color=COLORS["purple_light"])),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _display_options_results(results: dict, ohlcv, initial_capital: float, strategy_type: str):
    """Render rich results panel for options backtest."""
    st.markdown("---")
    strat_label = _OPTIONS_STRATEGIES.get(strategy_type, (strategy_type,))[0]
    section_header("📈", f"Options Backtest Results — {strat_label}", "COMPLETE")

    pv = results.get("portfolio_value", [])
    tr = results.get("total_return", 0) or 0
    cagr = results.get("cagr", 0) or 0
    dd = results.get("max_drawdown", 0) or 0
    sharpe = results.get("sharpe", 0) or 0
    trades_log = results.get("trades", [])

    end_cap = pv[-1] if pv else initial_capital
    net_pl = end_cap - initial_capital
    net_pct = net_pl / max(initial_capital, 1) * 100

    win_rate = None
    if trades_log:
        wins = sum(1 for t in trades_log if t.get("pnl_$", 0) > 0)
        win_rate = wins / len(trades_log) * 100

    kpi_row([
        ("Start Capital", format_currency(initial_capital), None, "neutral"),
        ("End Capital", format_currency(end_cap),
         f"{net_pct:+.1f}%", "positive" if net_pl >= 0 else "negative"),
        ("Net P/L", f"${net_pl:+,.0f}", None, "positive" if net_pl >= 0 else "negative"),
        ("CAGR", f"{cagr * 100:+.1f}%", None, "positive" if cagr > 0 else "negative"),
        ("Sharpe", f"{sharpe:.2f}", None, "neutral"),
        ("Max DD", f"{dd * 100:.1f}%", None, "negative" if dd < 0 else "neutral"),
    ])

    if win_rate is not None:
        st.markdown(
            f'<div style="text-align:center;color:#8b8b9e;font-size:0.8rem;margin-top:4px;">'
            f'Win rate: <strong style="color:#a855f7;">{win_rate:.1f}%</strong> '
            f'over {len(trades_log)} trades</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    if not pv or len(pv) < 3:
        st.warning("Insufficient data to render charts. Try a longer date range.")
        return

    tab_eq, tab_dd, tab_dist, tab_vs, tab_log, tab_snap = st.tabs([
        "Equity Curve", "Drawdown", "Returns Dist",
        "vs Buy & Hold", "Trade Log", "Options Snapshot",
    ])

    with tab_eq:
        fig = build_equity_curve(pv, initial_capital)
        st.plotly_chart(fig, use_container_width=True, key="opt_equity")

    with tab_dd:
        fig = build_drawdown_chart(pv)
        st.plotly_chart(fig, use_container_width=True, key="opt_dd")

    with tab_dist:
        fig = build_returns_distribution(pv)
        st.plotly_chart(fig, use_container_width=True, key="opt_dist")

    with tab_vs:
        if ohlcv is not None and not ohlcv.empty:
            bh = initial_capital * (ohlcv["close"].values / ohlcv["close"].values[0])
            x_pv = list(range(len(pv)))
            x_bh = list(range(len(bh)))
            fig_vs = make_plotly_fig(height=360)
            fig_vs.add_trace(go.Scatter(
                x=x_pv, y=pv, mode="lines",
                name=f"Options ({strat_label})",
                line=dict(color=COLORS["purple"], width=2),
                hovertemplate="Day %{x}<br>$%{y:,.0f}<extra>Options</extra>",
            ))
            # Trim BH to same length as PV for alignment
            min_len = min(len(bh), len(pv))
            fig_vs.add_trace(go.Scatter(
                x=x_bh[:min_len], y=list(bh[:min_len]),
                mode="lines", name="Buy & Hold",
                line=dict(color=COLORS["orange"], width=2, dash="dot"),
                hovertemplate="Day %{x}<br>$%{y:,.0f}<extra>B&H</extra>",
            ))
            fig_vs.add_hline(y=initial_capital, line_color="rgba(148,163,184,0.25)",
                             line_dash="dot")
            fig_vs.update_layout(
                title=dict(text="Options Strategy vs Buy & Hold Underlying",
                           font=dict(size=13, color=COLORS["purple_light"])),
                yaxis_title="Portfolio Value ($)",
                xaxis_title="Trading Days",
            )
            st.plotly_chart(fig_vs, use_container_width=True, key="opt_vs_bh")
        else:
            st.info("Underlying data not available for comparison.")

    with tab_log:
        if trades_log:
            df_trades = pd.DataFrame(trades_log)
            # Colour wins green, losses red
            def _color_pnl(val):
                color = "#22c55e" if val > 0 else "#ef4444"
                return f"color: {color}"
            st.dataframe(
                df_trades.style.applymap(_color_pnl, subset=["pnl_$"]),
                use_container_width=True, hide_index=True,
            )
            if trades_log:
                avg_hold = np.mean([t["days_held"] for t in trades_log])
                avg_pnl = np.mean([t["pnl_$"] for t in trades_log])
                st.caption(
                    f"Avg hold: **{avg_hold:.1f}d** | Avg P&L/trade: **${avg_pnl:+,.2f}**"
                )
        else:
            st.info("No discrete trade log available for this simulation mode.")

    with tab_snap:
        snap = results.get("options_snapshot")
        if snap:
            st.markdown(f"""
            <div class="phi-glass-card">
                <div style="color:#a855f7; font-weight:700; font-size:0.95rem; margin-bottom:14px;">
                    Live Chain Snapshot — Delta Anchor
                </div>
                <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:14px;">
                    <div><div class="phi-kpi-label">Source</div>
                         <div style="color:#e8e8ed;">{snap.get('source','—')}</div></div>
                    <div><div class="phi-kpi-label">Strike</div>
                         <div style="color:#e8e8ed;">${snap.get('strike',0):.2f}</div></div>
                    <div><div class="phi-kpi-label">Expiry</div>
                         <div style="color:#e8e8ed;">{snap.get('expiry','—')}</div></div>
                    <div><div class="phi-kpi-label">Mid Price</div>
                         <div style="color:#e8e8ed;">${snap.get('mid',0):.2f}</div></div>
                    <div><div class="phi-kpi-label">Delta</div>
                         <div style="color:#e8e8ed;">{snap.get('delta','—')}</div></div>
                    <div><div class="phi-kpi-label">Implied Vol</div>
                         <div style="color:#e8e8ed;">
                             {f"{snap.get('implied_volatility',0)*100:.1f}%"
                              if snap.get('implied_volatility') else '—'}
                         </div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(
                "No live chain snapshot available. "
                "Set **MARKETDATAAPP_API_TOKEN** in `.env` to anchor delta from a real options chain."
            )


# ─────────────────────────────────────────────────────────────────────────
# GEX SURFACE HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _augment_chain_gamma(chain_df: pd.DataFrame, spot: float, r: float = 0.05) -> pd.DataFrame:
    """Add Black-Scholes gamma to a yfinance chain (which omits greeks by default)."""
    df = chain_df.copy()
    today = date.today()
    try:
        from scipy.stats import norm as _norm
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    gamma_out = []
    for _, row in df.iterrows():
        try:
            K = float(row.get("strike", spot))
            iv = float(row.get("impliedvolatility", 0.25) or 0.25)
            iv = np.clip(iv, 0.01, 5.0)
            exp_str = str(row.get("expiration", ""))
            dte = max(1, (pd.Timestamp(exp_str).date() - today).days) if exp_str else 30
            T = dte / 365.0
            if _have_scipy and spot > 0 and K > 0 and T > 0:
                d1 = (np.log(spot / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
                gamma = float(_norm.pdf(d1) / (spot * iv * np.sqrt(T)))
            else:
                gamma = 0.0
        except Exception:
            gamma = 0.0
        gamma_out.append(gamma)

    df["gamma"] = gamma_out
    return df


def _build_gex_strike_chart(gex_series: pd.Series, spot: float, features: dict) -> go.Figure:
    """
    Horizontal bar chart of dealer GEX by strike.
    Green = dealers long gamma (pinning). Red = dealers short gamma (amplifying).
    """
    strikes = gex_series.index.astype(float).values
    gex_vals = gex_series.values.astype(float)
    max_abs = float(np.abs(gex_vals).max()) if len(gex_vals) else 1.0
    gex_norm = gex_vals / (max_abs + 1e-10)

    colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in gex_norm]

    fig = make_plotly_fig(height=520)
    fig.add_trace(go.Bar(
        x=gex_norm, y=strikes, orientation="h",
        marker_color=colors, opacity=0.80, name="Net Dealer GEX",
        customdata=gex_vals,
        hovertemplate="Strike: $%{y:.2f}<br>GEX (norm): %{x:.4f}<br>Raw GEX: %{customdata:.4e}<extra></extra>",
    ))

    # Spot
    fig.add_hline(y=spot, line_color=COLORS["purple"], line_dash="dash", line_width=2,
                  annotation_text=f"Spot ${spot:.2f}",
                  annotation_font_color=COLORS["purple_light"],
                  annotation_position="bottom right")

    # Gamma walls (highest |GEX| above & below spot)
    for mask, label, apos in [
        (strikes >= spot, "Wall ▲", "top right"),
        (strikes < spot,  "Wall ▼", "bottom right"),
    ]:
        sub_k = strikes[mask]
        sub_g = np.abs(gex_norm[mask])
        if len(sub_k):
            wall_k = float(sub_k[np.argmax(sub_g)])
            fig.add_hline(y=wall_k, line_color=COLORS["yellow"], line_dash="dot", line_width=1.5,
                          annotation_text=f"Gamma {label}  ${wall_k:.0f}",
                          annotation_font_color=COLORS["yellow"],
                          annotation_position=apos)

    # GEX flip zone highlight
    if features.get("gex_flip_zone", 0) > 0:
        half_w = abs(features.get("gamma_wall_distance", 0.03)) * spot * 0.3 or spot * 0.015
        fig.add_hrect(y0=spot - half_w, y1=spot + half_w,
                      fillcolor="rgba(249,115,22,0.08)", line_width=0,
                      annotation_text="GEX Flip Zone",
                      annotation_font_color=COLORS["orange"])

    fig.update_layout(
        title=dict(text="Dealer GEX by Strike  ·  Green=pinning, Red=amplifying",
                   font=dict(size=13, color=COLORS["purple_light"])),
        xaxis=dict(title="Normalised GEX", zeroline=True,
                   zerolinecolor="rgba(148,163,184,0.3)"),
        yaxis=dict(title="Strike ($)", tickformat="$.0f"),
        bargap=0.08,
    )
    return fig


def _build_gex_term_structure(chain_df: pd.DataFrame) -> go.Figure:
    """Bar chart of total |GEX| concentration by expiry (term structure)."""
    df = chain_df.copy()
    df.columns = [c.lower() for c in df.columns]
    today = date.today()

    if "expiration" not in df.columns:
        fig = make_plotly_fig(height=260)
        fig.add_annotation(text="No expiration data", showarrow=False,
                           font=dict(color="#8b8b9e", size=14))
        return fig

    df["_dte"] = df["expiration"].astype(str).map(
        lambda s: max(0, (pd.Timestamp(s).date() - today).days)
        if s not in ("", "nan", "None") else 999
    )
    df = df[df["_dte"] <= 180].copy()

    oi_col = next((c for c in ("openinterest", "volume") if c in df.columns), None)
    gamma_col = "gamma" if "gamma" in df.columns else None

    if gamma_col and oi_col:
        df["_weight"] = (
            pd.to_numeric(df[gamma_col], errors="coerce").fillna(0).clip(lower=0)
            * pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
        )
    elif oi_col:
        df["_weight"] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
    else:
        df["_weight"] = 1.0

    agg = df.groupby("_dte")["_weight"].sum().sort_index()
    if agg.empty:
        return make_plotly_fig(height=260)

    dom_dte = int(agg.idxmax())
    bar_colors = [
        COLORS["orange"] if int(d) == dom_dte else COLORS["purple"]
        for d in agg.index
    ]

    fig = make_plotly_fig(height=280)
    fig.add_trace(go.Bar(
        x=[f"{d}d" for d in agg.index], y=agg.values,
        marker_color=bar_colors, opacity=0.85, name="GEX Weight",
        hovertemplate="DTE: %{x}<br>Weight: %{y:.2e}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"GEX Term Structure  ·  Dominant expiry: <b>{dom_dte}d</b> (orange)",
                   font=dict(size=13, color=COLORS["purple_light"])),
        xaxis_title="Days to Expiry", yaxis_title="Total GEX Weight",
        bargap=0.25,
    )
    return fig


def _render_gex_surface_section(symbol: str, spot: float) -> None:
    """Full GEX Surface sub-panel rendered inside the Options Lab."""
    section_header("⚡", "GEX Surface — Dealer Gamma Exposure")

    st.caption(
        "**Positive GEX** (green) → dealers net-long gamma → _absorb_ moves → "
        "mean-reversion / pinning.  "
        "**Negative GEX** (red) → dealers net-short gamma → _amplify_ moves → trending / volatile.  "
        "The **GEX flip point** (zero-crossing) is the critical regime-change level."
    )

    fetch_col, info_col = st.columns([1, 3])
    with fetch_col:
        fetch_btn = st.button(
            f"⬇  Fetch Live Chain ({symbol})", key="gex_fetch", use_container_width=True
        )
    with info_col:
        st.caption("Uses yfinance options chain. BS gamma is computed per-contract since yfinance "
                   "does not supply greeks. For liquid underlyings (SPY, QQQ, AAPL, etc.) "
                   "this gives an accurate GEX landscape.")

    if fetch_btn:
        with st.spinner(f"Fetching {symbol} options chain…"):
            try:
                import yfinance as yf
                tk = yf.Ticker(symbol)
                exps = tk.options
                if not exps:
                    st.error(f"No options data for {symbol}.")
                    return
                frames = []
                for exp in exps[:6]:     # limit to near-term expirations
                    try:
                        chain = tk.option_chain(exp)
                    except Exception:
                        continue
                    for leg_df, ot in [(chain.calls, "call"), (chain.puts, "put")]:
                        tmp = leg_df.copy()
                        tmp["optiontype"] = ot
                        tmp["expiration"] = exp
                        frames.append(tmp)

                if not frames:
                    st.error("No chain data returned.")
                    return

                raw = pd.concat(frames, ignore_index=True)
                raw.columns = [c.lower() for c in raw.columns]
                chain_aug = _augment_chain_gamma(raw, spot)
                st.session_state["gex_chain"] = chain_aug
                st.session_state["gex_symbol"] = symbol
                st.session_state["gex_spot"] = spot
                st.success(f"Loaded {len(chain_aug):,} contracts across {len(exps[:6])} expirations.")
            except Exception as exc:
                st.error(f"Chain fetch failed: {exc}")
                return

    chain_df = st.session_state.get("gex_chain")
    cached_sym = st.session_state.get("gex_symbol", "")

    if chain_df is None or cached_sym != symbol:
        st.info(
            f"Press **Fetch Live Chain ({symbol})** to load the chain and render "
            "the GEX surface, gamma walls, and term structure."
        )
        return

    spot_used = float(st.session_state.get("gex_spot", spot))

    # ── Run GammaSurface engine ──────────────────────────────────────────
    with st.spinner("Computing GEX profile…"):
        try:
            from regime_engine.gamma_surface import GammaSurface
            gs = GammaSurface(config={
                "kernel_width_pct": 0.005,
                "min_oi": 10,
                "max_dte": 180,
                "gex_flip_threshold": 0.05,
            })
            features = gs.compute_features(chain_df, spot_used)
            gex_raw = gs._compute_gex_profile(chain_df, spot_used)
            gex_smoothed = gs._smooth_surface(gex_raw, spot_used) if not gex_raw.empty else gex_raw
        except Exception as exc:
            st.warning(f"GammaSurface: {exc}")
            features = {"gamma_net": 0, "gamma_wall_distance": 0,
                        "gamma_expiry_days": 30, "gex_flip_zone": 0}
            gex_smoothed = pd.Series(dtype=float)

    # ── KPI cards ────────────────────────────────────────────────────────
    gn  = float(features.get("gamma_net", 0))
    gwd = float(features.get("gamma_wall_distance", 0))
    ged = float(features.get("gamma_expiry_days", 30))
    gfz = float(features.get("gex_flip_zone", 0))

    regime_label = ("PINNING" if gn > 0.1 else "AMPLIFYING" if gn < -0.1 else "NEUTRAL")
    regime_color = (COLORS["green"] if gn > 0.1 else
                    COLORS["red"]   if gn < -0.1 else COLORS["yellow"])

    kpi_c = st.columns(5)
    for col, (lbl, val, clr) in zip(kpi_c, [
        ("Gamma Regime",    regime_label,                                regime_color),
        ("Net GEX at Spot", f"{gn:+.4f}",                               COLORS["green"] if gn >= 0 else COLORS["red"]),
        ("Wall Distance",   f"{gwd:+.2%}",                              COLORS["cyan"]),
        ("Dominant Expiry", f"{ged:.0f}d",                              COLORS["orange"]),
        ("GEX Flip Zone",   "ACTIVE ⚠" if gfz > 0 else "Clear",        COLORS["red"] if gfz > 0 else COLORS["text_muted"]),
    ]):
        with col:
            st.markdown(
                f'<div class="phi-kpi-card" style="border-top:2px solid {clr}44;">'
                f'<div class="phi-kpi-label">{lbl}</div>'
                f'<div style="color:{clr};font-weight:700;font-size:0.9rem;margin-top:4px;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Charts ───────────────────────────────────────────────────────────
    if gex_smoothed.empty:
        st.warning("Insufficient chain data for GEX profile. Try a more liquid symbol (SPY, QQQ, AAPL).")
    else:
        gex_col, ts_col = st.columns([3, 2])
        with gex_col:
            st.plotly_chart(
                _build_gex_strike_chart(gex_smoothed, spot_used, features),
                use_container_width=True, key="gex_strike_chart",
            )
        with ts_col:
            st.plotly_chart(
                _build_gex_term_structure(chain_df),
                use_container_width=True, key="gex_term_chart",
            )

    # ── Interpretation guide ─────────────────────────────────────────────
    with st.expander("Reading the GEX Surface", expanded=False):
        st.markdown(f"""
**Current regime: <span style="color:{regime_color}">{regime_label}</span>**

| GEX Condition | Dealer Position | Market Behaviour | Strategy Bias |
|---|---|---|---|
| GEX > 0 (green) | Long gamma | Buy dips, sell rips → **pinning** | Sell premium: Iron Condor, Covered Call |
| GEX < 0 (red) | Short gamma | Chase moves to hedge → **amplifying** | Buy directional: long calls/puts, straddle |
| GEX flip point | Zero-crossing | Regime change level — most critical price | Watch for breakout/breakdown |
| Gamma wall ▲/▼ | Peak \\|GEX\\| strike | Strong support/resistance magnet | Key strike for spreads / hedges |
| Dominant expiry | Peak term GEX | OpEx driving current gamma dynamics | Roll before expiry to avoid gamma collapse |

**Wall distance** {gwd:+.2%} from spot →
{"price is **{:.0f}% above** a major gamma wall (possible pin)".format(abs(gwd)*100) if gwd > 0
 else "price is **{:.0f}% below** a major gamma wall".format(abs(gwd)*100)}.
        """, unsafe_allow_html=True)

    # ── Raw chain table ───────────────────────────────────────────────────
    with st.expander(f"Raw Options Chain — {symbol} ({len(chain_df):,} contracts)", expanded=False):
        show_cols = [c for c in
                     ["strike", "expiration", "optiontype", "bid", "ask",
                      "impliedvolatility", "openinterest", "gamma", "volume"]
                     if c in chain_df.columns]
        st.dataframe(
            chain_df[show_cols].sort_values(["expiration", "strike"]).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


def _display_engine_results(results: dict, ohlcv, initial_capital: float, symbol: str) -> None:
    """Render results of the Full Engine Backtest."""
    st.markdown("---")
    section_header("⚡", "Engine Backtest Results", "FULL ENGINE")

    pv   = results.get("portfolio_value", [])
    tr   = results.get("total_return", 0) or 0
    cagr = results.get("cagr", 0) or 0
    dd   = results.get("max_drawdown", 0) or 0
    sh   = results.get("sharpe", 0) or 0
    wr   = results.get("win_rate", 0) or 0
    nt   = results.get("n_trades", 0) or 0

    end_cap = pv[-1] if pv else initial_capital
    net_pl  = end_cap - initial_capital

    # Engine component status
    flags = {
        "RegimeEngine": results.get("_regime_engine_ok", False),
        "GammaSurface": results.get("_gamma_ok", False),
        "OptionsEngine": results.get("_options_ok", False),
    }
    flag_html = "  ".join(
        f'<span style="color:{"#22c55e" if ok else "#ef4444"}; font-size:0.75rem;">'
        f'{"●" if ok else "○"} {name}</span>'
        for name, ok in flags.items()
    )
    st.markdown(f'<div style="margin-bottom:10px;">{flag_html}</div>', unsafe_allow_html=True)

    kpi_row([
        ("Start Capital",  format_currency(initial_capital),    None,   "neutral"),
        ("End Capital",    format_currency(end_cap),
         f"{(net_pl/max(initial_capital,1))*100:+.1f}%",
         "positive" if net_pl >= 0 else "negative"),
        ("CAGR",           f"{cagr*100:+.1f}%",                None,   "positive" if cagr > 0 else "negative"),
        ("Sharpe",         f"{sh:.2f}",                         None,   "neutral"),
        ("Max DD",         f"{dd*100:.1f}%",                    None,   "negative" if dd < 0 else "neutral"),
        ("Win Rate",       f"{wr*100:.1f}%",                    f"{nt} trades",  "positive" if wr > 0.5 else "neutral"),
    ])

    st.markdown("")

    if not pv or len(pv) < 3:
        st.warning("Not enough data points to render charts. Try a longer date range or lower the lookback.")
        return

    # Structures + regimes breakdown
    structs = results.get("structures_used", {})
    regimes = results.get("regimes_at_entry", {})

    if structs or regimes:
        mix_col, reg_col = st.columns(2)
        with mix_col:
            if structs:
                fig_s = go.Figure(go.Bar(
                    x=list(structs.keys()), y=list(structs.values()),
                    marker_color=COLORS["purple"], opacity=0.85,
                    hovertemplate="%{x}: %{y} trades<extra></extra>",
                ))
                fig_s.update_layout(
                    **PLOTLY_LAYOUT, height=260,
                    title=dict(text="Structures Selected by Engine",
                               font=dict(size=13, color=COLORS["purple_light"])),
                    xaxis_title="Structure", yaxis_title="# Trades",
                )
                st.plotly_chart(fig_s, use_container_width=True, key="eng_structs")
        with reg_col:
            if regimes:
                fig_r = go.Figure(go.Bar(
                    x=list(regimes.keys()), y=list(regimes.values()),
                    marker_color=COLORS["orange"], opacity=0.85,
                    hovertemplate="%{x}: %{y} trades<extra></extra>",
                ))
                fig_r.update_layout(
                    **PLOTLY_LAYOUT, height=260,
                    title=dict(text="Dominant Regime at Trade Entry",
                               font=dict(size=13, color=COLORS["purple_light"])),
                    xaxis_title="Regime", yaxis_title="# Trades",
                )
                st.plotly_chart(fig_r, use_container_width=True, key="eng_regimes")

    # Main chart tabs
    tab_eq, tab_dd, tab_dist, tab_vs, tab_log = st.tabs([
        "Equity Curve", "Drawdown", "Returns Dist", "vs Buy & Hold", "Trade Log",
    ])

    with tab_eq:
        st.plotly_chart(build_equity_curve(pv, initial_capital),
                        use_container_width=True, key="eng_equity")

    with tab_dd:
        st.plotly_chart(build_drawdown_chart(pv),
                        use_container_width=True, key="eng_dd")

    with tab_dist:
        st.plotly_chart(build_returns_distribution(pv),
                        use_container_width=True, key="eng_dist")

    with tab_vs:
        if ohlcv is not None and not ohlcv.empty:
            bh = initial_capital * (ohlcv["close"].values / ohlcv["close"].values[0])
            fig_vs = make_plotly_fig(height=360)
            fig_vs.add_trace(go.Scatter(
                x=list(range(len(pv))), y=pv, mode="lines",
                name="Engine Backtest",
                line=dict(color=COLORS["purple"], width=2),
                hovertemplate="Day %{x}<br>$%{y:,.0f}<extra>Engine</extra>",
            ))
            min_len = min(len(bh), len(pv))
            fig_vs.add_trace(go.Scatter(
                x=list(range(min_len)), y=list(bh[:min_len]),
                mode="lines", name="Buy & Hold",
                line=dict(color=COLORS["orange"], width=2, dash="dot"),
                hovertemplate="Day %{x}<br>$%{y:,.0f}<extra>B&H</extra>",
            ))
            fig_vs.add_hline(y=initial_capital, line_color="rgba(148,163,184,0.25)",
                             line_dash="dot")
            fig_vs.update_layout(
                title=dict(text=f"Engine ({symbol}) vs Buy & Hold",
                           font=dict(size=13, color=COLORS["purple_light"])),
                yaxis_title="Portfolio Value ($)", xaxis_title="Trading Days",
            )
            st.plotly_chart(fig_vs, use_container_width=True, key="eng_vs_bh")

    with tab_log:
        trades = results.get("trades", [])
        if trades:
            df_t = pd.DataFrame(trades)
            display_cols = [c for c in [
                "entry_date", "exit_date", "structure", "level",
                "regime", "vol_regime", "gex_regime", "confidence",
                "entry_spot", "exit_spot", "entry_iv", "days_held",
                "exit_reason", "pnl_$", "cum_ret_%",
            ] if c in df_t.columns]

            def _color_pnl(val):
                return f"color: {'#22c55e' if val > 0 else '#ef4444'}"

            styled = df_t[display_cols].style.applymap(_color_pnl, subset=["pnl_$"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Exit reason breakdown
            reason_counts = df_t["exit_reason"].value_counts()
            st.caption(
                "Exit reasons — "
                + "  |  ".join(f"**{r}**: {c}" for r, c in reason_counts.items())
            )
        else:
            st.info(
                "No trades recorded. The engine may need a longer date range, "
                "lower confidence gate, or the RegimeEngine may be unavailable."
            )


def page_options_backtest():
    """Full-featured Options Lab — strategy builder, Greeks dashboard, payoff diagram, backtest."""
    section_header("📊", "Options Lab — Strategy Backtester", "OPTIONS")

    st.caption(
        "Design multi-leg options strategies, preview payoff diagrams and live Greeks, "
        "then run a full Black-Scholes–based backtest over historical data."
    )

    # ── Step 1: Dataset & Strategy ───────────────────────────────────────
    with st.expander("Step 1 — Symbol, Date Range & Capital", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            opt_sym = st.text_input("Underlying Symbol", value="SPY", key="opt_sym")
        with c2:
            opt_start = st.date_input("Start Date", value=date(2020, 1, 1), key="opt_start")
        with c3:
            opt_end = st.date_input("End Date", value=date(2024, 12, 31), key="opt_end")
        with c4:
            opt_capital = st.number_input(
                "Capital ($)", value=100_000, min_value=1_000, step=10_000, key="opt_capital"
            )

    # ── Step 2: Strategy & Parameters ───────────────────────────────────
    with st.expander("Step 2 — Strategy Selection & Parameters", expanded=True):
        strat_col, desc_col = st.columns([1, 2])
        with strat_col:
            opt_strategy = st.selectbox(
                "Strategy Type",
                options=list(_OPTIONS_STRATEGIES.keys()),
                format_func=lambda k: f"{_OPTIONS_STRATEGIES[k][1]}  {_OPTIONS_STRATEGIES[k][0]}",
                key="opt_strategy",
            )
        with desc_col:
            sname, sicon, sdesc = _OPTIONS_STRATEGIES[opt_strategy]
            st.markdown(f"""
            <div class="phi-glass-card" style="padding:14px; margin-top:4px;">
                <div style="color:#a855f7; font-weight:700; font-size:1rem;">{sicon} {sname}</div>
                <div style="color:#8b8b9e; font-size:0.85rem; margin-top:6px;">{sdesc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        p1, p2, p3 = st.columns(3)
        with p1:
            opt_delta = st.slider(
                "Delta Assumption", 0.10, 0.90, 0.50, 0.05, key="opt_delta",
                help="ATM ≈ 0.50 | OTM ≈ 0.25-0.35 | Deep ITM ≈ 0.70-0.85",
            )
            opt_iv = st.slider(
                "Implied Volatility (%)", 5, 120, 25, 1, key="opt_iv",
                help="Annual IV used for Black-Scholes pricing",
            ) / 100.0
        with p2:
            opt_pos_pct = st.slider(
                "Position Size (% Capital)", 1, 50, 10, 1, key="opt_pos_pct",
                help="Notional deployed per trade as % of portfolio",
            ) / 100.0
            opt_dte = st.slider(
                "Days to Expiry (DTE)", 7, 365, 30, 1, key="opt_dte",
                help="Option contract length; position re-entered after each cycle",
            )
        with p3:
            opt_profit = st.slider(
                "Profit Target (%)", 10, 200, 50, 5, key="opt_profit",
                help="Exit when position gains this % of the initial premium",
            ) / 100.0
            opt_stop = st.slider(
                "Stop Loss (%)", 10, 100, 30, 5, key="opt_stop",
                help="Exit when position loses this % of the initial premium",
            ) / 100.0

    # ── Greeks Dashboard + Payoff Preview ──────────────────────────────
    section_header("🔢", "Live Greeks & Payoff Diagram")
    st.caption("Greeks and diagram update instantly from the parameters above.")

    @st.cache_data(ttl=3600, show_spinner=False)
    def _get_spot_price(symbol: str) -> float:
        end_d = date.today()
        start_d = end_d - timedelta(days=14)
        df = _load_ohlcv_yf(symbol, str(start_d), str(end_d))
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return 500.0

    spot_price = _get_spot_price(opt_sym)
    T_years = max(opt_dte, 1) / 365.0
    R_RATE = 0.05

    # Compute composite Greeks for the selected strategy
    opt_type_primary = "call" if opt_strategy in (
        "long_call", "bull_call_spread", "covered_call"
    ) else "put"

    raw_greeks = _bs_greeks(spot_price, spot_price, T_years, R_RATE, opt_iv, opt_type_primary)

    if opt_strategy == "iron_condor":
        gc = _bs_greeks(spot_price, spot_price * 1.05, T_years, R_RATE, opt_iv, "call")
        gp = _bs_greeks(spot_price, spot_price * 0.95, T_years, R_RATE, opt_iv, "put")
        raw_greeks = {
            "delta": -(gc["delta"] + gp["delta"]),
            "gamma": -(gc["gamma"] + gp["gamma"]),
            "theta": -(gc["theta"] + gp["theta"]),
            "vega":  -(gc["vega"]  + gp["vega"]),
        }
    elif opt_strategy == "straddle":
        gc = _bs_greeks(spot_price, spot_price, T_years, R_RATE, opt_iv, "call")
        gp = _bs_greeks(spot_price, spot_price, T_years, R_RATE, opt_iv, "put")
        raw_greeks = {k: gc[k] + gp[k] for k in gc}

    atm_call_px = _bs_price(spot_price, spot_price, T_years, R_RATE, opt_iv, "call")
    atm_put_px  = _bs_price(spot_price, spot_price, T_years, R_RATE, opt_iv, "put")
    _, net_cost = _strategy_legs(opt_strategy, spot_price, T_years, R_RATE, opt_iv)

    greek_cols = st.columns(6)
    greek_items = [
        ("Spot Price", f"${spot_price:,.2f}", "#c084fc"),
        ("Delta (Δ)", f"{raw_greeks['delta']:+.4f}", COLORS["green"] if raw_greeks["delta"] >= 0 else COLORS["red"]),
        ("Gamma (Γ)", f"{raw_greeks['gamma']:.6f}", COLORS["cyan"]),
        ("Theta (Θ)", f"{raw_greeks['theta']:+.4f}/d", COLORS["yellow"]),
        ("Vega (ν)", f"{raw_greeks['vega']:+.4f}", COLORS["orange"]),
        ("Net Premium", f"${abs(net_cost):.2f} {'debit' if net_cost > 0 else 'credit'}", COLORS["purple"]),
    ]
    for col, (label, value, color) in zip(greek_cols, greek_items):
        with col:
            st.markdown(f"""
            <div class="phi-kpi-card" style="border-top:2px solid {color}33;">
                <div class="phi-kpi-label">{label}</div>
                <div style="color:{color}; font-weight:700; font-size:1rem; margin-top:4px;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # Payoff diagram + IV surface side by side
    diag_col, iv_col = st.columns([3, 2])
    with diag_col:
        payoff_fig = build_payoff_diagram(opt_strategy, spot_price, opt_iv, opt_dte, R_RATE)
        st.plotly_chart(payoff_fig, use_container_width=True, key="opt_payoff")
    with iv_col:
        iv_surface_fig = _build_iv_surface_fig(spot_price, opt_iv, R_RATE)
        st.plotly_chart(iv_surface_fig, use_container_width=True, key="opt_iv_surface")

    # Black-Scholes pricing matrix
    with st.expander("Black-Scholes Pricing Matrix", expanded=False):
        st.caption("ATM call & put prices across a range of IVs and DTEs.")
        iv_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
        dte_range = [7, 14, 30, 60, 90, 180]
        matrix_data = {}
        for iv_val in iv_range:
            row = {}
            for dte_val in dte_range:
                T_m = dte_val / 365.0
                call_p = _bs_price(spot_price, spot_price, T_m, R_RATE, iv_val, "call")
                put_p  = _bs_price(spot_price, spot_price, T_m, R_RATE, iv_val, "put")
                row[f"{dte_val}d"] = f"${call_p:.2f}C / ${put_p:.2f}P"
            matrix_data[f"{iv_val*100:.0f}% IV"] = row
        st.dataframe(
            pd.DataFrame(matrix_data).T,
            use_container_width=True,
        )

    # ── GEX Surface ──────────────────────────────────────────────────────
    st.markdown("---")
    _render_gex_surface_section(opt_sym.strip().upper(), spot_price)

    # ── Run Backtest ─────────────────────────────────────────────────────
    st.markdown("---")
    section_header("🚀", "Backtest")

    tab_simple, tab_engine = st.tabs([
        "Strategy Backtest (BS Sim)",
        "Full Engine Backtest (MFT + OptionsEngine)",
    ])

    # ── Tab 1: Strategy Backtest ──────────────────────────────────────────
    with tab_simple:
        st.caption(
            "Black-Scholes mark-to-market simulation for the selected strategy. "
            "Fast — does not run the MFT regime engine."
        )
        run_info = st.container()
        with run_info:
            info_c1, info_c2, info_c3, info_c4 = st.columns(4)
            with info_c1:
                st.markdown(f'<div class="phi-kpi-label">Symbol</div>'
                            f'<div style="color:#e8e8ed;font-weight:600;">{opt_sym.upper()}</div>',
                            unsafe_allow_html=True)
            with info_c2:
                st.markdown(f'<div class="phi-kpi-label">Period</div>'
                            f'<div style="color:#e8e8ed;font-weight:600;">{opt_start} → {opt_end}</div>',
                            unsafe_allow_html=True)
            with info_c3:
                st.markdown(f'<div class="phi-kpi-label">Strategy</div>'
                            f'<div style="color:#a855f7;font-weight:700;">{_OPTIONS_STRATEGIES[opt_strategy][0]}</div>',
                            unsafe_allow_html=True)
            with info_c4:
                st.markdown(f'<div class="phi-kpi-label">Capital</div>'
                            f'<div style="color:#e8e8ed;font-weight:600;">{format_currency(opt_capital)}</div>',
                            unsafe_allow_html=True)

        st.markdown("")
        if st.button("🚀 Run Strategy Backtest", type="primary", use_container_width=True, key="opt_run"):
            _execute_options_backtest_page(
                symbol=opt_sym.strip().upper(),
                start=str(opt_start), end=str(opt_end),
                capital=float(opt_capital),
                strategy_type=opt_strategy,
                delta=opt_delta, iv=opt_iv, dte=opt_dte,
                position_pct=opt_pos_pct,
                exit_profit=opt_profit, exit_stop=opt_stop,
            )

        if (
            st.session_state.get("opt_results")
            and st.session_state.get("opt_symbol_used", "") == opt_sym.strip().upper()
            and st.session_state.get("opt_strategy_used", "") == opt_strategy
        ):
            _display_options_results(
                results=st.session_state["opt_results"],
                ohlcv=st.session_state.get("opt_ohlcv"),
                initial_capital=float(st.session_state.get("opt_capital_used", opt_capital)),
                strategy_type=st.session_state.get("opt_strategy_used", opt_strategy),
            )
        elif st.session_state.get("opt_results"):
            st.info("Results above are from a previous run. Press **Run Strategy Backtest** to update.")

    # ── Tab 2: Full Engine Backtest ───────────────────────────────────────
    with tab_engine:
        st.markdown("""
        <div class="phi-glass-card" style="margin-bottom:16px;">
            <div style="color:#a855f7; font-weight:700; font-size:1rem; margin-bottom:8px;">
                ⚡ Full MFT + OptionsEngine Walk-Forward Backtest
            </div>
            <div style="color:#8b8b9e; font-size:0.85rem; line-height:1.6;">
                At each bar this runs the <strong>complete pipeline</strong>:<br>
                RegimeEngine → synthetic BS chain → GammaSurface → OptionsEngine.select_trade()<br>
                The engine autonomously picks the optimal structure (L1/L2/L3) based on
                regime probabilities, IV regime, and GEX. Positions are marked to market
                daily with Black-Scholes and exited on profit target / stop / DTE expiry.
            </div>
        </div>
        """, unsafe_allow_html=True)

        eng_c1, eng_c2, eng_c3 = st.columns(3)
        with eng_c1:
            eng_lookback = st.slider(
                "Regime lookback (bars)", 30, 200, 60, 10, key="eng_lookback",
                help="Rolling window fed to RegimeEngine at each bar",
            )
            eng_min_conf = st.slider(
                "Min confidence gate", 0.20, 0.80, 0.40, 0.05, key="eng_min_conf",
                help="OptionsEngine only trades above this confidence level",
            )
        with eng_c2:
            eng_dte = st.slider(
                "Target DTE (main leg)", 14, 90, 30, 1, key="eng_dte",
            )
            eng_dte_short = st.slider(
                "Front-month DTE", 7, 45, 14, 1, key="eng_dte_short",
                help="Used for calendar spreads and covered-call short leg",
            )
        with eng_c3:
            eng_pos_pct = st.slider(
                "Position size (% capital)", 1, 40, 10, 1, key="eng_pos_pct",
            ) / 100.0
            eng_iv_prem = st.slider(
                "IV premium multiplier", 1.00, 2.00, 1.15, 0.05, key="eng_iv_prem",
                help="ATM IV = realised vol × this multiplier (VIX > HV effect)",
            )

        eng_profit, eng_stop = st.columns(2)
        with eng_profit:
            eng_exit_profit = st.slider(
                "Profit target (%)", 10, 200, 50, 5, key="eng_exit_profit",
            ) / 100.0
        with eng_stop:
            eng_exit_stop = st.slider(
                "Stop loss (%)", 10, 100, 30, 5, key="eng_exit_stop",
            ) / 100.0

        st.markdown("")

        run_eng_btn = st.button(
            "⚡ Run Full Engine Backtest",
            type="primary", use_container_width=True, key="eng_run",
        )

        if run_eng_btn:
            with st.spinner("Fetching OHLCV data…"):
                eng_ohlcv = _load_ohlcv_yf(
                    opt_sym.strip().upper(), str(opt_start), str(opt_end)
                )
            if eng_ohlcv is None or eng_ohlcv.empty:
                st.error(f"No data for {opt_sym}.")
            else:
                prog = st.progress(0.0, text="Starting engine backtest…")

                def _update_prog(f):
                    prog.progress(
                        min(f, 1.0),
                        text=f"Engine backtest: {f*100:.0f}% complete…",
                    )

                with st.spinner("Running walk-forward engine backtest…"):
                    try:
                        from phi.options.engine_backtest import run_engine_backtest
                        eng_results = run_engine_backtest(
                            ohlcv=eng_ohlcv,
                            symbol=opt_sym.strip().upper(),
                            initial_capital=float(opt_capital),
                            position_pct=eng_pos_pct,
                            lookback_bars=eng_lookback,
                            min_confidence=eng_min_conf,
                            dte_days=eng_dte,
                            dte_short=eng_dte_short,
                            exit_profit_pct=eng_exit_profit,
                            exit_stop_pct=eng_exit_stop,
                            iv_premium=eng_iv_prem,
                            progress_cb=_update_prog,
                        )
                        prog.progress(1.0, text="Complete!")
                        st.session_state["eng_results"] = eng_results
                        st.session_state["eng_ohlcv"]   = eng_ohlcv
                        st.session_state["eng_sym"]      = opt_sym.strip().upper()
                    except Exception as exc:
                        prog.empty()
                        st.error(f"Engine backtest failed: {exc}")

        # ── Engine results display ────────────────────────────────────────
        eng_res = st.session_state.get("eng_results")
        if eng_res and st.session_state.get("eng_sym", "") == opt_sym.strip().upper():
            _display_engine_results(
                eng_res,
                st.session_state.get("eng_ohlcv"),
                float(opt_capital),
                opt_sym.strip().upper(),
            )
        elif eng_res:
            st.info("Engine results above are from a previous run. Press **Run Full Engine Backtest** to update.")


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR + NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
def build_sidebar():
    """Build the premium sidebar with navigation."""
    phi_logo_sidebar()

    st.sidebar.markdown(f"""
    <div style="display:flex; align-items:center; gap:6px; padding:8px 0;">
        {status_dot('live')}
        <span style="color:#8b8b9e; font-size:0.75rem;">Engine Online</span>
        <span style="color:#5a5a70; font-size:0.7rem; margin-left:auto;">
            {datetime.now().strftime('%H:%M:%S')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    pages = {
        "Command Center": "🏠",
        "Workbench": "🧪",
        "Options Lab": "📊",
        "MFT Blender": "🔬",
        "Scanner": "🔍",
        "AI Advisor": "🤖",
        "ML Models": "🧠",
        "System": "⚙",
    }

    current = st.session_state.get("active_page", "Command Center")

    for page_name, icon in pages.items():
        is_active = current == page_name
        btn_type = "primary" if is_active else "secondary"
        if st.sidebar.button(f"{icon}  {page_name}", key=f"nav_{page_name}",
                              use_container_width=True, type=btn_type):
            st.session_state["active_page"] = page_name
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding:12px 0; text-align:center;">
        <div style="color:#5a5a70; font-size:0.7rem; letter-spacing:0.05em;">
            PHI-NANCE PRO v2.0<br>
            <span style="color:#a855f7;">Market Field Theory Engine</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Phi-nance Pro",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()
    build_sidebar()

    # Route to active page
    page = st.session_state.get("active_page", "Command Center")

    PAGE_MAP = {
        "Command Center": page_command_center,
        "Workbench": page_workbench,
        "Options Lab": page_options_backtest,
        "MFT Blender": page_mft_blender,
        "Scanner": page_scanner,
        "AI Advisor": page_ai_advisor,
        "ML Models": page_ml_models,
        "System": page_system,
    }

    render_fn = PAGE_MAP.get(page, page_command_center)

    # Page title
    st.title("Phi-nance Pro")
    st.caption("Quantitative Trading Intelligence — Market Field Theory Engine")

    render_fn()

    # Footer
    st.markdown("""
    <div class="phi-footer">
        PHI-NANCE PRO &nbsp;·&nbsp; Quantitative Trading Intelligence &nbsp;·&nbsp;
        Market Field Theory &nbsp;·&nbsp; Built for Professional Traders
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

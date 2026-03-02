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
import threading
import importlib
import hashlib
import warnings
from pathlib import Path

# Suppress Lumibot pandas FutureWarning (Series.__getitem__)
warnings.filterwarnings("ignore", category=FutureWarning, module="lumibot.entities.bars")
from datetime import date, datetime, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
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
    cols = st.columns(4)
    with cols[0]:
        if st.button("🔍 Scan Market", use_container_width=True, key="cmd_scan"):
            st.session_state["active_page"] = "Scanner"
            st.rerun()
    with cols[1]:
        if st.button("🧪 Run Backtest", use_container_width=True, key="cmd_bt"):
            st.session_state["active_page"] = "Workbench"
            st.rerun()
    with cols[2]:
        if st.button("🔬 MFT Analysis", use_container_width=True, key="cmd_mft"):
            st.session_state["active_page"] = "MFT Blender"
            st.rerun()
    with cols[3]:
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

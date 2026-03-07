#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Live Backtest Workbench -- Premium Edition v3.1
==========================================================
SaaS-grade step-by-step quant workbench:
  - AUTO WEB/MOBILE DETECTION with adaptive layout
  - Animated step wizard with progress tracking
  - Fetch & cache historical data with premium previews
  - Select & tune indicators with visual cards
  - Blend multiple indicators with weight visualization
  - PhiAI auto-tuning with progress animation
  - Equities + Options backtests with rich result displays
  - Premium dark glassmorphism theme

Run:
    python -m streamlit run app_streamlit/live_workbench.py
"""

import os
import sys
import time
import threading
import importlib
import warnings
from pathlib import Path
from datetime import date, datetime

# Suppress Lumibot pandas FutureWarning (Series.__getitem__)
warnings.filterwarnings("ignore", category=FutureWarning, module="lumibot.entities.bars")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("IS_BACKTESTING", "True")

from app_streamlit.device_detect import detect_device, get_device, inject_responsive_meta, _JS_DETECT
from phi.utils.updater import UpdateManager

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Indicator catalog (maps to strategies)
# ─────────────────────────────────────────────────────────────────────────────
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (BrokenPipeError, OSError):
    pass


# ---------------------------------------------------------------------------
# Premium CSS + HTML Components
# ---------------------------------------------------------------------------
_CSS_PATH = _ROOT / ".streamlit" / "styles.css"


def _inject_css():
    """Inject premium CSS + JS/fonts via proper channels.
    
    st.markdown handles <style> fine but strips <script>/<link>.
    Use components.html (zero-height iframe) for JS and font loading.
    """
    import streamlit.components.v1 as components
    css = _CSS_PATH.read_text(encoding='utf-8') if _CSS_PATH.exists() else ''
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    components.html(_JS_DETECT, height=0)


def _render_kpi_row(kpis):
    cards = ""
    for label, value, delta, dtype in kpis:
        delta_html = f'<div class="phi-kpi-delta {dtype}">{delta}</div>' if delta else ""
        cards += f'''
        <div class="phi-kpi-card">
            <div class="phi-kpi-label">{label}</div>
            <div class="phi-kpi-value">{value}</div>
            {delta_html}
        </div>'''
    return f'<div class="phi-kpi-row">{cards}</div>'


def _render_section_header(icon, title, badge=""):
    badge_html = f'<span class="phi-section-badge">{badge}</span>' if badge else ""
    return f'''
    <div class="phi-section-header">
        <span class="phi-section-icon">{icon}</span>
        <span class="phi-section-title">{title}</span>
        {badge_html}
    </div>'''


def _render_signal_badge(signal):
    cls = {"BUY": "phi-signal-buy", "SELL": "phi-signal-sell"}.get(signal, "phi-signal-hold")
    return f'<span class="phi-signal {cls}">{signal}</span>'


def _render_status_dot(status):
    return f'<span class="phi-status-dot {status}"></span>'


def _render_step_indicator(steps):
    """Render a visual step progress indicator."""
    html = '<div class="phi-steps">'
    for i, (label, status) in enumerate(steps):
        num = i + 1
        html += f'''
        <div class="phi-step {status}">
            <span class="phi-step-number">{num}</span>
            <span>{label}</span>
        </div>'''
        if i < len(steps) - 1:
            html += '<div class="phi-step-connector"></div>'
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Plotly Premium Theme
# ---------------------------------------------------------------------------
PHI_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,18,26,0.3)",
    font=dict(color="#eeeef2", family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor="rgba(168,85,247,0.05)", showgrid=True),
    yaxis=dict(gridcolor="rgba(168,85,247,0.05)", showgrid=True),
    legend=dict(bgcolor="rgba(18,18,26,0.7)", bordercolor="rgba(168,85,247,0.12)", borderwidth=1),
    hoverlabel=dict(bgcolor="#1a1a26", bordercolor="#a855f7", font=dict(color="#eeeef2")),
)
CHART_COLORS = ["#a855f7", "#f97316", "#22c55e", "#06b6d4", "#eab308", "#ec4899", "#8b5cf6", "#14b8a6"]


def _phi_chart(fig, height=0):
    """Render a Plotly chart with device-aware height."""
    dev = get_device()
    if height <= 0:
        height = dev.chart_height
    elif dev.is_phone:
        height = max(200, int(height * 0.7))
    elif dev.is_tablet:
        height = max(240, int(height * 0.85))
    fig.update_layout(**PHI_LAYOUT, height=height, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": dev.is_desktop, "displaylogo": False,
        "scrollZoom": not dev.is_mobile,
    })


# ---------------------------------------------------------------------------
# Indicator Catalog
# ---------------------------------------------------------------------------
INDICATOR_CATALOG = {
    "RSI": {
        "description": "Relative Strength Index -- oversold/overbought signals.",
        "params": {"rsi_period": (2, 50, 14), "oversold": (10, 50, 30), "overbought": (50, 95, 70)},
        "strategy": "strategies.rsi.RSIStrategy",
        "icon": "&#x1F4C8;",
    },
    "MACD": {
        "description": "MACD crossover -- bullish/bearish momentum.",
        "params": {"fast_period": (2, 50, 12), "slow_period": (10, 100, 26), "signal_period": (2, 30, 9)},
        "strategy": "strategies.macd.MACDStrategy",
        "icon": "&#x26A1;",
    },
    "Bollinger": {
        "description": "Bollinger Bands -- buy below lower, sell above upper.",
        "params": {"bb_period": (5, 100, 20), "num_std": (1, 4, 2)},
        "strategy": "strategies.bollinger.BollingerBands",
        "icon": "&#x1F4CA;",
    },
    "Dual SMA": {
        "description": "Golden cross / death cross dual moving average.",
        "params": {"fast_period": (2, 100, 10), "slow_period": (10, 300, 50)},
        "strategy": "strategies.dual_sma.DualSMACrossover",
        "icon": "&#x2194;",
    },
    "Mean Reversion": {
        "description": "Buy below SMA, sell above -- classic mean reversion.",
        "params": {"sma_period": (5, 200, 20)},
        "strategy": "strategies.mean_reversion.MeanReversion",
        "icon": "&#x1F504;",
    },
    "Breakout": {
        "description": "Donchian channel breakout/breakdown.",
        "params": {"channel_period": (5, 100, 20)},
        "strategy": "strategies.breakout.ChannelBreakout",
        "icon": "&#x1F680;",
    },
    "Buy & Hold": {
        "description": "Naive long-only baseline.",
        "params": {},
        "strategy": "strategies.buy_and_hold.BuyAndHold",
        "icon": "&#x1F4B5;",
    },
    "VWAP": {
        "description": "Session VWAP deviation — intraday mean-reversion. Best on 1m–1H bars.",
        "params": {"band_pct": (0.1, 2.0, 0.5)},
        "strategy": None,  # direct backtest only; no Lumibot strategy wrapper
        "icon": "&#x23F1;",
    },
}

BLEND_METHODS = ["Weighted Sum", "Regime-Weighted", "Voting", "PhiAI Chooses"]
METRICS = ["ROI", "CAGR", "Sharpe", "Max Drawdown", "Direction Accuracy", "Profit Factor"]
EXIT_STRATEGIES = ["Signal exit", "SL/TP", "Trailing stop", "Time exit"]
POSITION_SIZING = ["Fixed %", "Fixed shares"]



def _inject_css():
    """Inject custom CSS from styles.css file."""
    _CSS_PATH = _ROOT / ".streamlit" / "styles.css"
    if _CSS_PATH.exists():
        with open(_CSS_PATH, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _sidebar():
    """Render the branded sidebar with navigation info."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1.5rem;">
                <div style="font-size:2rem; margin-bottom:0.3rem;">📊</div>
                <div style="font-size:1.3rem; font-weight:700;
                            background:linear-gradient(135deg,#a855f7,#f97316);
                            -webkit-background-clip:text;
                            -webkit-text-fill-color:transparent;
                            background-clip:text;">
                    Phi-nance
                </div>
                <div style="color:#71717a; font-size:0.75rem;
                            letter-spacing:0.08em; text-transform:uppercase;
                            margin-top:0.2rem;">
                    Live Backtest Workbench
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="color:#52525b; font-size:0.72rem;
                        text-transform:uppercase; letter-spacing:0.1em;
                        font-weight:600; margin-bottom:0.6rem; padding:0 0.2rem;">
                Workflow
            </div>
            """,
            unsafe_allow_html=True,
        )

        steps = [
            ("1", "Dataset Builder", "Fetch & cache OHLCV"),
            ("2", "Indicators", "Select & tune signals"),
            ("3", "Blending", "Combine multiple signals"),
            ("4", "PhiAI", "Auto-optimize everything"),
            ("5", "Backtest", "Run & review results"),
        ]
        for num, name, desc in steps:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.7rem;
                            padding:0.55rem 0.6rem; border-radius:8px;
                            margin-bottom:0.3rem;
                            border:1px solid rgba(168,85,247,0.08);
                            background:rgba(168,85,247,0.04);">
                    <div style="min-width:22px; height:22px;
                                background:linear-gradient(135deg,#a855f7,#7c3aed);
                                border-radius:50%; display:flex;
                                align-items:center; justify-content:center;
                                color:#fff; font-size:0.7rem; font-weight:700;
                                box-shadow:0 2px 6px rgba(168,85,247,0.4);">
                        {num}
                    </div>
                    <div>
                        <div style="color:#e4e4e7; font-size:0.82rem;
                                    font-weight:600; line-height:1.2;">{name}</div>
                        <div style="color:#71717a; font-size:0.72rem;">{desc}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="border-top:1px solid rgba(168,85,247,0.12);
                        padding-top:1rem; color:#52525b; font-size:0.72rem;
                        text-align:center; line-height:1.6;">
                Regime-aware &bull; Cached &bull; Reproducible<br>
                <span style="color:rgba(168,85,247,0.5);">&#9632;</span>
                Purple = signals &nbsp;
                <span style="color:rgba(249,115,22,0.5);">&#9632;</span>
                Orange = caution
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_update_banner() -> None:
    """Check tracked tool versions and render update notifications."""
    updater = UpdateManager(session_state=st.session_state)
    statuses = updater.check_all(force=False)
    if statuses:
        st.session_state["tool_updates"] = statuses

    updates = st.session_state.get("tool_updates_available", [])
    if not updates:
        return

    st.warning(f"Updates available for {len(updates)} external tool(s).")
    with st.expander("View available updates", expanded=False):
        for item in updates:
            st.write(
                f"- **{item.display_name}**: {item.current_version or 'unknown'} → {item.latest_version or 'unknown'}"
            )
def _section_header(num: str, title: str, subtitle: str = ""):
    """Render a styled step section header."""
    sub_html = (
        f'<div style="color:#71717a; font-size:0.82rem; '
        f'margin-top:0.2rem;">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:0.75rem;
                    margin: 1.8rem 0 0.8rem;">
            <div style="min-width:32px; height:32px;
                        background:linear-gradient(135deg,#a855f7,#7c3aed);
                        border-radius:50%; display:flex; align-items:center;
                        justify-content:center; color:#fff; font-size:0.85rem;
                        font-weight:700;
                        box-shadow:0 2px 10px rgba(168,85,247,0.45);">
                {num}
            </div>
            <div>
                <div style="color:#e4e4e7; font-size:1.1rem; font-weight:700;
                            letter-spacing:-0.01em;">{title}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _load_strategy(module_cls: str):
    """
    Dynamically load a strategy class from a string.
    """
def _load_strategy(module_cls):
    module_path, cls_name = module_cls.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _av_datasource():
    from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
    return AlphaVantageFixedDataSource


def _run_backtest(strategy_class, params, config):
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")
    tf = config.get("timeframe", "1D")
    timestep = "day" if tf == "1D" else "minute"
    symbol = config["symbols"][0]
    start = config["start"]
    end = config["end"]
    vendor = config.get("vendor", "yfinance")

    # ── Try to load cached OHLCV data first ──────────────────────────────────
    from phi.data import get_cached_dataset, fetch_and_cache
    df = get_cached_dataset(vendor, symbol, tf, str(start.date()), str(end.date()))

    if df is None or df.empty:
        # Auto-fetch if not cached (yfinance always works, no key needed)
        _api_key = {
            "finnhub":  os.getenv("FINNHUB_API_KEY"),
            "stockdata": os.getenv("STOCKDATA_API_KEY"),
            "massive":  os.getenv("MASSIVE_API_KEY"),
        }.get(vendor, os.getenv("AV_API_KEY"))
        try:
            df = fetch_and_cache(vendor, symbol, tf, str(start.date()), str(end.date()), api_key=_api_key)
        except Exception:
            df = None

    if df is not None and not df.empty:
        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # Build pandas_data dict for PandasDataBacktesting
        asset = Asset(symbol, asset_type=Asset.AssetTypes.STOCK)
        try:
            from lumibot.backtesting import PandasDataBacktesting
            pandas_data = {asset: {"df": df, "timestep": timestep}}
            results, strat = strategy_class.run_backtest(
                datasource_class=PandasDataBacktesting,
                backtesting_start=start,
                backtesting_end=end,
                budget=config["initial_capital"],
                benchmark_asset=config.get("benchmark", symbol),
                parameters=params,
                pandas_data=pandas_data,
                show_plot=False,
                show_tearsheet=False,
                save_tearsheet=False,
                show_indicators=False,
                show_progress_bar=False,
                quiet_logs=True,
            )
            return results, strat
        except Exception as e:
            # PandasDataBacktesting failed — log and fall through to AV
            print(f"PandasDataBacktesting failed ({e}), falling back to AV data source")

    # ── Fallback: AlphaVantageFixedDataSource ────────────────────────────────
    from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")
    results, strat = strategy_class.run_backtest(
        datasource_class=_av_datasource(),
        backtesting_start=config["start"], backtesting_end=config["end"],
        budget=config["initial_capital"], benchmark_asset=config.get("benchmark", "SPY"),
        parameters=params, api_key=av_api_key, timestep=timestep,
        show_plot=False, show_tearsheet=False, save_tearsheet=False,
        show_indicators=False, show_progress_bar=False, quiet_logs=True,
    )
    return results, strat


def _compute_accuracy(strat):
    from strategies.prediction_tracker import compute_prediction_accuracy
    return compute_prediction_accuracy(strat)


def _extract_scalar(val):
    if isinstance(val, dict):
        for key in ("drawdown", "value", "max_drawdown", "return"):
            if key in val: return val[key]
        for v in val.values():
            if isinstance(v, (int, float)): return v
        return None
    return val


def _run_fully_automated(symbol, start_date, end_date, capital, use_ollama, ollama_host):
    progress = st.progress(0, text="Fully automated: initializing pipeline...")
    result_holder, exc_holder = [None], [None]

    def run():
        try:
            from phi.phiai.auto_pipeline import run_fully_automated as run_pipeline
            cfg, indicators, blend_method, explanation, ohlcv = run_pipeline(
                symbol=symbol, start_date=start_date, end_date=end_date,
                initial_capital=capital, ollama_host=ollama_host, use_ollama=use_ollama,
            )
            blend_weights = {k: 1.0 / len(indicators) for k in indicators}
            from phi.backtest import run_direct_backtest
            results, strat = run_direct_backtest(
                ohlcv=ohlcv, symbol=symbol, indicators=indicators,
                blend_weights=blend_weights, blend_method=blend_method,
                signal_threshold=0.15, initial_capital=capital,
            )
            sc = _compute_accuracy(strat) if hasattr(strat, "prediction_log") else {}
            result_holder[0] = (cfg, results, strat, indicators, blend_method, sc, explanation)
        except Exception as e:
            exc_holder[0] = e

    th = threading.Thread(target=run)
    th.start()
    pct, start_t = 10, time.time()
    stages = ["Fetching data...", "Analyzing features...", "AI optimization...",
              "Building strategy...", "Running backtest...", "Finalizing..."]
    while th.is_alive():
        time.sleep(0.4)
        elapsed = time.time() - start_t
        pct = min(95, 10 + int(elapsed * 1.2))
        stage_idx = min(len(stages) - 1, int(elapsed / 5))
        progress.progress(pct / 100, text=f"{stages[stage_idx]} {pct}%")

    if exc_holder[0]:
        progress.empty()
        st.error(str(exc_holder[0]))
        st.exception(exc_holder[0])
        return

    progress.progress(1.0, text="Complete!")
    time.sleep(0.5)
    progress.empty()

    cfg, results, strat, indicators, blend_method, sc, explanation = result_holder[0]
    blend_weights = {k: 1.0 / len(indicators) for k in indicators}

    with st.expander("AI Decisions", expanded=True):
        st.markdown(f"""
        <div style="background:rgba(168,85,247,0.04);border:1px solid rgba(168,85,247,0.12);
                    border-radius:12px;padding:1.2rem;font-size:0.88rem;line-height:1.6;
                    color:#b0b0c0;font-family:Inter,sans-serif;">
            {explanation.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    _display_results(cfg, results, strat, indicators, blend_method, blend_weights, sc)


# ---------------------------------------------------------------------------
# Step 1 -- Dataset Builder
# ---------------------------------------------------------------------------
def render_dataset_builder():
    """Render Step 1: Data fetching and caching UI."""
    _section_header("1", "Dataset Builder", "Fetch & cache OHLCV data from your chosen vendor")
    dev = get_device()
    st.markdown(_render_section_header("", "STEP 1 -- DATASET BUILDER", "DATA"), unsafe_allow_html=True)

    if dev.is_phone:
        trading_mode = st.selectbox("Trading Mode", ["Equities", "Options"], key="ds_mode")
        symbols_raw = st.text_input("Symbol(s)", value="SPY", key="ds_symbols", help="Comma-separated: SPY, QQQ, AAPL")
        start_d = st.date_input("Start", value=date(2020, 1, 1), key="ds_start")
        end_d = st.date_input("End", value=date(2024, 12, 31), key="ds_end")
        timeframe = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m", "5m", "1m"], key="ds_tf")
        vendor = st.selectbox("Data Vendor", ["Alpha Vantage", "yfinance", "Binance Public"], key="ds_vendor")
        initial_capital = st.number_input("Initial Capital ($)", value=100_000, min_value=1_000, step=10_000, key="ds_cap")
    else:
        col_mode, col_sym, col_range = st.columns([1, 2, 2])
        with col_mode:
            trading_mode = st.selectbox("Trading Mode", ["Equities", "Options"], key="ds_mode")
        with col_sym:
            symbols_raw = st.text_input("Symbol(s)", value="SPY", key="ds_symbols", help="Comma-separated: SPY, QQQ, AAPL")
        with col_range:
            start_d = st.date_input("Start", value=date(2020, 1, 1), key="ds_start")
            end_d = st.date_input("End", value=date(2024, 12, 31), key="ds_end")

        col_tf, col_vendor, col_cap = st.columns(3)
        with col_tf:
            timeframe = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m", "5m", "1m"], key="ds_tf")
        with col_vendor:
            vendor = st.selectbox("Data Vendor", ["Alpha Vantage", "yfinance", "Binance Public"], key="ds_vendor")
        with col_cap:
            initial_capital = st.number_input("Initial Capital ($)", value=100_000, min_value=1_000, step=10_000, key="ds_cap")

    if start_d >= end_d:
        st.error("Start date must be before end date.")
        return None
    if initial_capital <= 0:
        st.error("Initial capital must be > 0")
        return None

    c1, c2 = st.columns(2)
    fetch_clicked = c1.button("Fetch & Cache", type="primary", key="ds_fetch", use_container_width=True)
    use_cached = c2.button("Use Cached", key="ds_use", use_container_width=True)

    vendor_map = {"Alpha Vantage": "alphavantage", "yfinance": "yfinance", "Binance Public": "binance_public"}
    vendor_key = vendor_map.get(vendor, "alphavantage")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        st.error("Enter at least one symbol.")
        return None

    dfs = {}
    if fetch_clicked or use_cached:
        from phi.data import fetch_and_cache, get_cached_dataset

        with st.status("Loading data...", expanded=True):
            for sym in symbols:
                try:
                    df = fetch_and_cache(vendor_key, sym, timeframe, str(start_d), str(end_d)) if fetch_clicked else get_cached_dataset(vendor_key, sym, timeframe, str(start_d), str(end_d))
                    if df is not None and not df.empty:
                        dfs[sym] = df
                except Exception as e:
                    st.error(f"{sym}: {e}")

        if dfs:
            st.session_state["workbench_dataset"] = dfs
            st.session_state["workbench_config"] = {
                "trading_mode": trading_mode.lower(),
                "symbols": symbols,
                "start": datetime.combine(start_d, datetime.min.time()),
                "end": datetime.combine(end_d, datetime.min.time()),
                "timeframe": timeframe,
                "vendor": vendor_key,
                "initial_capital": float(initial_capital),
            }
            st.success(f"Loaded {len(dfs)} dataset(s).")

    cfg = st.session_state.get("workbench_config")
    if cfg:
        st.caption(f"Current dataset: {', '.join(cfg['symbols'])} | {cfg['timeframe']} | {cfg['vendor']}")
    return cfg


# ---------------------------------------------------------------------------
# Step 2 -- Indicator Selection
# ---------------------------------------------------------------------------
def render_indicator_selection():
    """Render Step 2: Strategy indicator selection and manual tuning."""
    _section_header("2", "Indicator Selection", "Choose and tune trading signals")

    dev = get_device()
    st.markdown(_render_section_header("", "STEP 2 -- INDICATORS", "SELECTION"), unsafe_allow_html=True)
    selected = st.session_state.get("workbench_indicators", {})

    search = st.text_input("Search indicators", key="ind_search", placeholder="RSI, MACD...")
    available = [k for k in INDICATOR_CATALOG if not search or search.lower() in k.lower()]

    n_ind_cols = dev.cols_strategy
    cols = st.columns(min(n_ind_cols, len(available)) if available else 1)
    for idx, name in enumerate(available):
        info = INDICATOR_CATALOG[name]
        with cols[idx % len(cols)]:
            with st.container(border=True):
                enabled = st.checkbox(f"**{name}**", value=name in selected, key=f"ind_{name}")
                st.caption(info["description"][:60])
                if enabled:
                    if name not in selected:
                        selected[name] = {"enabled": True, "auto_tune": False, "params": {}}
                    selected[name]["enabled"] = True
                    selected[name]["auto_tune"] = st.toggle("PhiAI Auto-tune",
                        value=selected[name].get("auto_tune", False), key=f"at_{name}")
                    with st.expander("Params"):
                        for pname, (lo, hi, default) in info["params"].items():
                            selected[name]["params"][pname] = st.slider(
                                pname, lo, hi, default, key=f"param_{name}_{pname}")
                else:
                    if name in selected:
                        del selected[name]

    st.session_state["workbench_indicators"] = selected
    if len(selected) > 4:
        st.warning(
            "⚠️ More than 4 indicators selected — this may slow down results significantly. "
            "Consider reducing scope for faster performance."
        )

    if selected:
        names = ', '.join(selected.keys())
        st.markdown(f"""
        <div class="phi-info-bar">
            <span class="phi-info-bar-label">Selected:</span>
            <span class="phi-info-bar-value">{names}</span>
        </div>
        """, unsafe_allow_html=True)

    return selected


# ---------------------------------------------------------------------------
# Step 3 -- Blending
# ---------------------------------------------------------------------------
def render_blending(indicators):
    dev = get_device()
    if len(indicators) < 2:
        st.caption("Select 2+ indicators to enable blending.")
        return "weighted_sum", {}

    _section_header("3", "Blending Panel", "Combine multiple signal streams")
    st.markdown(_render_section_header("", "STEP 3 -- BLENDING", "MULTI-SIGNAL"), unsafe_allow_html=True)

    method = st.selectbox("Blend Mode", BLEND_METHODS, key="blend_method")
    method_map = {"Weighted Sum": "weighted_sum", "Regime-Weighted": "regime_weighted",
                  "Voting": "voting", "PhiAI Chooses": "phiai_chooses"}
    method_key = method_map.get(method, "weighted_sum")

    weights = {}
    blend_n_cols = 2 if dev.is_phone else min(len(indicators), 4)
    cols = st.columns(blend_n_cols)
    for idx, name in enumerate(indicators):
        with cols[idx % blend_n_cols]:
            weights[name] = st.slider(f"{name}", 0.0, 1.0, round(1.0 / len(indicators), 2), 0.05, key=f"wt_{name}")

    # Blend weight visualization
    if weights:
        total = sum(weights.values())
        if total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.55,
                marker=dict(colors=CHART_COLORS[:len(weights)],
                            line=dict(color='#12121a', width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, color="#eeeef2"),
                hovertemplate="%{label}: %{value:.2f} (%{percent})<extra></extra>",
            )])
            fig.update_layout(
                title=dict(text="BLEND WEIGHTS", font=dict(size=12, color="#7a7a90")),
                showlegend=False, height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#eeeef2"),
            )
            st.plotly_chart(fig, use_container_width=True)

    return method_key, weights


# ---------------------------------------------------------------------------
# Step 4 -- PhiAI
# ---------------------------------------------------------------------------
def render_phiai():
    """
    Render the PhiAI panel for automated optimization.
    """
    _section_header("4", "PhiAI Panel", "Regime-aware automated optimization")
    phiai_full = st.toggle(
        "PhiAI Full Auto", value=False, key="phiai_full",
        help="Auto-enable/disable indicators, tune params, select blend"
    )
    st.markdown(_render_section_header("", "STEP 4 -- PHI-AI", "AUTO-OPTIMIZE"), unsafe_allow_html=True)
    phiai_full = st.toggle("PhiAI Full Auto", value=False, key="phiai_full",
                            help="Auto-enable/disable indicators, tune params, select blend")
    if phiai_full:
        st.markdown("""
        <div class="phi-info-bar" style="border-color:rgba(168,85,247,0.25);">
            <span style="color:#c084fc;font-size:0.85rem;">
                PhiAI will optimize indicators, parameters, and blend with regime-aware adjustments.
            </span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.number_input("Max indicators", 1, 10, 5, key="phiai_max")
        c2.checkbox("No shorting", value=True, key="phiai_noshort")
    return phiai_full


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Backtest Controls
# ─────────────────────────────────────────────────────────────────────────────
def render_backtest_controls(config: dict):
    """
    Render the backtest control panel based on trading mode.
    """
    if not config:
        return {}

    _section_header("5", "Backtest Controls", "Position sizing and exit rules")
# ---------------------------------------------------------------------------
# Step 5 -- Backtest Controls
# ---------------------------------------------------------------------------
def render_backtest_controls(config):
    dev = get_device()
    if not config: return {}
    st.markdown(_render_section_header("", "STEP 5 -- CONTROLS",
                config.get("trading_mode", "equities").upper()), unsafe_allow_html=True)
    mode = config.get("trading_mode", "equities")

    if mode == "equities":
        if dev.is_phone:
            allow_short = st.checkbox("Allow shorting", value=False, key="bt_short")
            pos_sizing = st.selectbox("Position sizing", POSITION_SIZING, key="bt_pos")
            exit_strat = st.selectbox("Exit strategy", EXIT_STRATEGIES, key="bt_exit")
            return {"allow_short": allow_short, "position_sizing": pos_sizing, "exit_strategy": exit_strat}
        c1, c2, c3 = st.columns(3)
        allow_short = c1.checkbox("Allow shorting", value=False, key="bt_short")
        pos_sizing = c2.selectbox("Position sizing", POSITION_SIZING, key="bt_pos")
        exit_strat = c3.selectbox("Exit strategy", EXIT_STRATEGIES, key="bt_exit")
        return {"allow_short": allow_short, "position_sizing": pos_sizing, "exit_strategy": exit_strat}
    else:
        st.warning("Options mode currently supports a single symbol and a single European option contract.")
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True, key="opt_type")
        c1, c2, c3 = st.columns(3)
        strike = c1.number_input("Strike", min_value=0.01, value=100.0, step=1.0, key="opt_strike")
        expiry = c2.date_input("Expiry", value=config.get("end", datetime.now()).date(), key="opt_expiry")
        multiplier = c3.number_input("Multiplier", min_value=1, value=100, step=1, key="opt_multiplier")

        c4, c5 = st.columns(2)
        iv = c4.number_input("Implied Volatility", min_value=0.1, max_value=1.0, value=0.3, step=0.01, key="opt_iv")
        r = c5.number_input("Risk-Free Rate", min_value=0.0, max_value=0.2, value=0.02, step=0.005, key="opt_r")

        opts = {
            "option_type": option_type,
            "strike": float(strike),
            "expiry": expiry,
            "multiplier": int(multiplier),
            "iv": float(iv),
            "r": float(r),
            "quantity": 1,
        }
        st.session_state["bt_options_controls"] = opts
        return opts


# ---------------------------------------------------------------------------
# Phibot Review helpers
# ---------------------------------------------------------------------------
_VERDICT_COLORS = {
    "strong":   "#22c55e",
    "moderate": "#f59e0b",
    "weak":     "#f97316",
    "neutral":  "#6b7280",
}
_CONFIDENCE_COLORS = {
    "high":   "#22c55e",
    "medium": "#f59e0b",
    "low":    "#6b7280",
}
_CATEGORY_LABELS = {
    "blend_weight":      "Blend Weight",
    "signal_threshold":  "Signal Threshold",
    "blend_method":      "Blend Method",
    "add_indicator":     "Add Indicator",
    "position_sizing":   "Position Sizing",
}


def _apply_phibot_tweaks(tweaks) -> int:
    """
    Apply adopted tweaks to session state so sliders / selectors update
    when Streamlit reruns. Returns count of tweaks applied.
    """
    applied = 0
    for tweak in tweaks:
        if not st.session_state.get(f"phibot_adopt_{tweak.id}", False):
            continue
        cat = tweak.category
        if cat == "blend_weight":
            st.session_state[tweak.param_key] = tweak.suggested_value
        elif cat == "blend_method":
            st.session_state["blend_method"] = tweak.suggested_value
        elif cat == "signal_threshold":
            st.session_state["phibot_signal_threshold"] = tweak.suggested_value
        elif cat == "add_indicator":
            ind_dict = dict(st.session_state.get("workbench_indicators", {}))
            if tweak.suggested_value and tweak.suggested_value not in ind_dict:
                ind_dict[tweak.suggested_value] = {"params": {}}
                st.session_state["workbench_indicators"] = ind_dict
        elif cat == "position_sizing":
            st.session_state["phibot_position_size"] = tweak.suggested_value
        # Reset the checkbox
        st.session_state[f"phibot_adopt_{tweak.id}"] = False
        applied += 1
    return applied


def _render_phibot_review(config, results, strat, indicators, blend_method, blend_weights):
    """Render the Phibot post-backtest review tab."""
    try:
        from phi.phibot.reviewer import review_backtest
    except Exception as exc:
        st.error(f"Phibot reviewer unavailable: {exc}")
        return

    # OHLCV from session state
    ohlcv = None
    try:
        sym  = (config.get("symbols") or ["SPY"])[0]
        dfs  = st.session_state.get("workbench_dataset", {})
        ohlcv = dfs.get(sym)
    except Exception:
        pass

    prediction_log = []
    if strat and hasattr(strat, "prediction_log"):
        prediction_log = strat.prediction_log or []

    try:
        review = review_backtest(
            ohlcv=ohlcv,
            results=results if isinstance(results, dict) else {},
            prediction_log=prediction_log,
            indicators=indicators or {},
            blend_weights=blend_weights or {},
            blend_method=blend_method or "weighted_sum",
            config=config or {},
        )
    except Exception as exc:
        st.error(f"Phibot analysis failed: {exc}")
        return

    # Persist tweaks in session state so _apply_phibot_tweaks can read them
    st.session_state["phibot_current_tweaks"] = review.tweaks

    # ── Header card ──────────────────────────────────────────────────────────
    verdict_color = _VERDICT_COLORS.get(review.verdict, "#6b7280")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.9rem;padding:1.1rem 1.4rem;
                    background:rgba(168,85,247,0.07);border-radius:12px;
                    border:1px solid rgba(168,85,247,0.2);margin-bottom:1.4rem;">
            <div style="font-size:1.8rem;line-height:1;">&#129302;</div>
            <div style="flex:1;">
                <div style="color:#c084fc;font-size:0.62rem;text-transform:uppercase;
                            letter-spacing:0.14em;font-weight:700;">Phibot Analysis</div>
                <div style="color:#e4e4e7;font-size:0.92rem;font-weight:500;
                            margin-top:0.25rem;line-height:1.45;">{review.summary}</div>
            </div>
            <div style="flex-shrink:0;">
                <span style="background:{verdict_color};color:#fff;font-size:0.62rem;
                             font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                             padding:0.35rem 0.85rem;border-radius:20px;">
                    {review.verdict}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Regime performance table ──────────────────────────────────────────────
    qualified = {r: s for r, s in review.regime_stats.items() if s.get("count", 0) >= 1}
    if qualified:
        st.markdown(
            "<div style='color:#7a7a90;font-size:0.68rem;text-transform:uppercase;"
            "letter-spacing:0.1em;font-weight:600;margin-bottom:0.6rem;'>"
            "Regime Performance</div>",
            unsafe_allow_html=True,
        )
        regime_rows = []
        for reg, s in sorted(qualified.items(), key=lambda x: -x[1]["count"]):
            from phi.phibot.reviewer import _REGIME_DESCRIPTIONS
            regime_rows.append(
                {
                    "Regime":     reg,
                    "Description": _REGIME_DESCRIPTIONS.get(reg, ""),
                    "Trades":     s["count"],
                    "Win Rate":   f"{s['win_rate']:.0%}",
                    "Avg P&L":    f"{s['avg_pl']:+.1%}",
                }
            )
        st.dataframe(
            pd.DataFrame(regime_rows),
            use_container_width=True,
            hide_index=True,
        )

    # ── Observations ─────────────────────────────────────────────────────────
    if review.observations:
        st.markdown(
            "<div style='color:#7a7a90;font-size:0.68rem;text-transform:uppercase;"
            "letter-spacing:0.1em;font-weight:600;margin:1.2rem 0 0.5rem;'>"
            "Key Observations</div>",
            unsafe_allow_html=True,
        )
        for obs in review.observations:
            st.markdown(f"- {obs}")

    # ── Tweaks ───────────────────────────────────────────────────────────────
    if not review.tweaks:
        st.markdown(
            "<div style='color:#a1a1aa;font-size:0.85rem;margin-top:1rem;'>"
            "No specific tweaks recommended — the strategy is well-configured for "
            "current market conditions.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        "<div style='color:#7a7a90;font-size:0.68rem;text-transform:uppercase;"
        "letter-spacing:0.1em;font-weight:600;margin:1.4rem 0 0.8rem;'>"
        "Suggested Tweaks</div>",
        unsafe_allow_html=True,
    )

    adopted_any = False
    for tweak in review.tweaks:
        conf_color = _CONFIDENCE_COLORS.get(tweak.confidence, "#6b7280")
        cat_label  = _CATEGORY_LABELS.get(tweak.category, tweak.category.replace("_", " ").title())

        # Format the change arrow
        if tweak.current_value is not None:
            change_txt = f"{tweak.current_value} &rarr; {tweak.suggested_value}"
        else:
            change_txt = f"Add: {tweak.suggested_value}"

        st.markdown(
            f"""
            <div style="background:rgba(249,115,22,0.05);border:1px solid rgba(249,115,22,0.18);
                        border-radius:10px;padding:0.9rem 1.1rem;margin-bottom:0.6rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:0.35rem;">
                    <span style="color:#fb923c;font-size:0.62rem;text-transform:uppercase;
                                 letter-spacing:0.1em;font-weight:700;">{cat_label}</span>
                    <span style="background:{conf_color}22;color:{conf_color};font-size:0.6rem;
                                 font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
                                 padding:0.2rem 0.55rem;border-radius:10px;">
                        {tweak.confidence} confidence
                    </span>
                </div>
                <div style="color:#e4e4e7;font-size:0.9rem;font-weight:600;
                            margin-bottom:0.3rem;">{tweak.title}</div>
                <div style="color:#a1a1aa;font-size:0.8rem;line-height:1.5;
                            margin-bottom:0.4rem;">{tweak.rationale}</div>
                <div style="color:#6b7280;font-size:0.75rem;
                            font-family:'JetBrains Mono',monospace;">{change_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        is_adopted = st.checkbox(
            "Adopt this tweak",
            key=f"phibot_adopt_{tweak.id}",
            value=st.session_state.get(f"phibot_adopt_{tweak.id}", False),
        )
        if is_adopted:
            adopted_any = True

    # ── Apply button ─────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    if adopted_any:
        if st.button(
            "Apply Adopted Tweaks & Re-run",
            key="phibot_apply_tweaks",
            type="primary",
            use_container_width=True,
        ):
            n = _apply_phibot_tweaks(review.tweaks)
            st.session_state["phibot_tweaks_applied"] = n
            st.rerun()
    else:
        st.caption("Check one or more tweaks above to enable the Apply button.")


# ---------------------------------------------------------------------------
# Results — helper builders
# ---------------------------------------------------------------------------

def _monthly_returns_pivot(prediction_log: list, portfolio_values: list) -> pd.DataFrame:
    """Return a year × month pivot of monthly returns, or empty DataFrame."""
    if not prediction_log or len(portfolio_values) < 10:
        return pd.DataFrame()
    try:
        dates = [r.get("date") for r in prediction_log if r.get("date") is not None]
        pvs = portfolio_values[1 : len(dates) + 1]
        if len(dates) < 5 or len(pvs) < 5:
            return pd.DataFrame()
        series = pd.Series(pvs, index=pd.DatetimeIndex(dates)).sort_index()
        try:
            monthly = series.resample("ME").last()
        except ValueError:
            monthly = series.resample("M").last()
        monthly_ret = monthly.pct_change().dropna()
        if len(monthly_ret) < 2:
            return pd.DataFrame()
        _M = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        df = pd.DataFrame({
            "year": monthly_ret.index.year,
            "month": monthly_ret.index.month,
            "ret": monthly_ret.values,
        })
        pivot = df.pivot(index="year", columns="month", values="ret")
        pivot.columns = [_M.get(c, str(c)) for c in pivot.columns]
        return pivot
    except Exception:
        return pd.DataFrame()


def _trade_statistics(prediction_log: list) -> dict:
    """Reconstruct trade P&L from the bar-by-bar signal log."""
    if not prediction_log:
        return {}
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    for i, row in enumerate(prediction_log):
        sig = row.get("signal", "NEUTRAL")
        price = float(row.get("price") or 0)
        prev_sig = prediction_log[i - 1].get("signal", "NEUTRAL") if i > 0 else "NEUTRAL"
        if sig == "UP" and prev_sig != "UP" and not in_trade and price > 0:
            in_trade, entry_price, entry_idx = True, price, i
        elif sig in ("DOWN", "NEUTRAL") and prev_sig == "UP" and in_trade and price > 0:
            in_trade = False
            trades.append({"pnl": price - entry_price, "bars": i - entry_idx})
    if not trades:
        return {}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_loss = abs(sum(losses))
    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_bars_held": float(np.mean([t["bars"] for t in trades])),
        "profit_factor": abs(sum(wins)) / gross_loss if gross_loss > 0 else float("inf"),
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "largest_win": float(max(wins)) if wins else 0.0,
        "largest_loss": float(min(losses)) if losses else 0.0,
    }


def _make_equity_fig(prediction_log: list, portfolio_values: list, cap: float):
    """Equity curve with buy ▲ / sell ▼ entry markers."""
    pvs = portfolio_values[1 : len(prediction_log) + 1] if prediction_log else portfolio_values[1:]
    if len(pvs) < 2:
        return None
    has_dates = bool(prediction_log and prediction_log[0].get("date") is not None)
    x_vals = [r.get("date") for r in prediction_log[: len(pvs)]] if has_dates else list(range(len(pvs)))
    line_color = "#22c55e" if pvs[-1] > cap else "#ef4444"
    fill_color = "rgba(34,197,94,0.05)" if pvs[-1] > cap else "rgba(239,68,68,0.05)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=pvs, mode="lines",
        line=dict(color=line_color, width=2, shape="spline", smoothing=0.3),
        fill="tozeroy", fillcolor=fill_color, name="Portfolio",
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    if prediction_log:
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        for i, (row, pv) in enumerate(zip(prediction_log[: len(pvs)], pvs)):
            sig = row.get("signal")
            prev = prediction_log[i - 1].get("signal") if i > 0 else "NEUTRAL"
            xi = row.get("date", i) if has_dates else i
            if sig == "UP" and prev != "UP":
                buy_x.append(xi); buy_y.append(pv)
            elif sig != "UP" and prev == "UP":
                sell_x.append(xi); sell_y.append(pv)
        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y, mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", size=9, color="#22c55e",
                            line=dict(color="#15803d", width=1)),
                hovertemplate="BUY $%{y:,.0f}<extra></extra>",
            ))
        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y, mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", size=9, color="#ef4444",
                            line=dict(color="#b91c1c", width=1)),
                hovertemplate="SELL $%{y:,.0f}<extra></extra>",
            ))
    fig.add_hline(y=cap, line_dash="dot", line_color="rgba(148,163,184,0.2)",
                  annotation_text=f"Start ${cap:,.0f}", annotation_font_color="#7a7a90",
                  annotation_font_size=10)
    fig.update_layout(
        yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=30, b=30),
    )
    return fig


def _make_drawdown_fig(portfolio_values: list):
    """Underwater drawdown area chart."""
    if not portfolio_values or len(portfolio_values) < 2:
        return None
    pv = np.array(portfolio_values, dtype=float)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / (peak + 1e-12) * 100
    fig = go.Figure(go.Scatter(
        y=dd, mode="lines",
        line=dict(color="rgba(239,68,68,0.7)", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
        name="Drawdown", hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="Drawdown", yaxis_ticksuffix="%",
        margin=dict(l=50, r=20, t=10, b=30),
    )
    return fig


def _make_monthly_heatmap_fig(pivot_df: pd.DataFrame):
    """Calendar heatmap of monthly returns — red/green diverging."""
    if pivot_df.empty:
        return None
    z = pivot_df.values * 100
    abs_max = float(max(np.nanmax(np.abs(z)), 1.0))
    text_vals = [[f"{v:+.1f}%" if not np.isnan(v) else "" for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=list(pivot_df.columns), y=[str(y) for y in pivot_df.index],
        text=text_vals, texttemplate="%{text}",
        textfont=dict(size=11, family="'JetBrains Mono','Courier New',monospace", color="#eeeef2"),
        colorscale=[
            [0.0,  "#7f1d1d"], [0.2, "#b91c1c"], [0.45, "#3d0d0d"],
            [0.5,  "#131318"],
            [0.55, "#0d3d0d"], [0.8, "#15803d"], [1.0,  "#14532d"],
        ],
        zmid=0, zmin=-abs_max, zmax=abs_max, showscale=False,
        hovertemplate="<b>%{y} %{x}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=10),
    )
    return fig


def _make_annual_returns_fig(pivot_df: pd.DataFrame):
    """Bar chart of approximate annual returns from monthly pivot."""
    if pivot_df.empty:
        return None
    annual = pivot_df.sum(axis=1) * 100
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in annual.values]
    fig = go.Figure(go.Bar(
        x=[str(y) for y in annual.index], y=annual.values,
        marker_color=colors, marker_opacity=0.85,
        text=[f"{v:+.1f}%" for v in annual.values], textposition="outside",
        textfont=dict(size=11, family="'JetBrains Mono',monospace"),
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis_ticksuffix="%", showlegend=False,
        margin=dict(l=40, r=20, t=20, b=30),
    )
    return fig


def _make_blend_fig(indicators: dict, blend_weights: dict):
    """Horizontal weight bars for active indicators."""
    enabled = [k for k, v in indicators.items()
               if isinstance(v, dict) and v.get("enabled", True)]
    if not enabled:
        return None
    total = sum(blend_weights.get(k, 1.0) for k in enabled) or 1.0
    pcts = [blend_weights.get(k, 1.0) / total * 100 for k in enabled]
    fig = go.Figure(go.Bar(
        y=enabled, x=pcts, orientation="h",
        marker=dict(color=CHART_COLORS[: len(enabled)], opacity=0.85),
        text=[f"{p:.0f}%" for p in pcts], textposition="inside",
        textfont=dict(size=11),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_ticksuffix="%", yaxis=dict(autorange="reversed"),
        showlegend=False, margin=dict(l=10, r=20, t=10, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
def _display_results(config, results, strat, indicators, blend_method, blend_weights, sc=None):
    if results is None:
        st.error("Backtest produced no results.")
        return
    sc = sc or {}

    # ── Extract scalars ────────────────────────────────────────────────────
    tr     = float(_extract_scalar(getattr(results, "total_return", None) or results.get("total_return")) or 0)
    cagr   = float(_extract_scalar(getattr(results, "cagr",         None) or results.get("cagr"))         or 0)
    dd     = float(_extract_scalar(getattr(results, "max_drawdown", None) or results.get("max_drawdown")) or 0)
    sharpe = float(_extract_scalar(getattr(results, "sharpe",       None) or results.get("sharpe"))       or 0)
    cap    = float(config.get("initial_capital", 100_000))
    pv     = list(getattr(results, "portfolio_value", None) or results.get("portfolio_value") or [])
    end_cap = float(pv[-1]) if pv else cap
    net_pl  = end_cap - cap
    acc     = float(sc.get("accuracy", 0) or 0)

    plog    = (strat.prediction_log if strat and hasattr(strat, "prediction_log") else []) or []
    tstats  = _trade_statistics(plog)
    monthly = _monthly_returns_pivot(plog, pv)

    # ── Derived display values ────────────────────────────────────────────
    symbol     = (config.get("symbols") or ["--"])[0]
    tf         = config.get("timeframe", "1D")
    start_lbl  = str(config.get("start", ""))[:10]
    end_lbl    = str(config.get("end",   ""))[:10]
    n_ind      = len([v for v in indicators.values()
                      if isinstance(v, dict) and v.get("enabled", True)])
    blend_lbl  = blend_method.replace("_", " ").title()
    pl_color   = "#22c55e" if net_pl >= 0 else "#ef4444"
    ret_color  = "#22c55e" if tr   >= 0 else "#ef4444"
    sh_color   = "#22c55e" if sharpe > 1 else ("#f97316" if sharpe > 0 else "#ef4444")
    win_rate   = tstats.get("win_rate", 0)
    pf         = tstats.get("profit_factor", 0)
    n_trades   = tstats.get("total_trades", 0)
    pf_str     = f"{pf:.2f}" if pf != float("inf") else "∞"

    opt_metrics = ""
    if n_trades > 0:
        opt_metrics += f"""
        <div class="phi-rm-item">
          <div class="phi-rm-value">{win_rate:.0%}</div>
          <div class="phi-rm-label">Win Rate</div>
        </div>
        <div class="phi-rm-item">
          <div class="phi-rm-value">{pf_str}</div>
          <div class="phi-rm-label">Profit Factor</div>
        </div>"""

    # ── Results banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="phi-results-banner">
      <div class="phi-results-meta-row">
        <span class="phi-results-symbol">{symbol}</span>
        <span class="phi-meta-sep">◆</span>
        <span class="phi-results-period">{start_lbl} → {end_lbl}</span>
        <span class="phi-meta-sep">◆</span>
        <span class="phi-results-tf">{tf}</span>
        <span class="phi-meta-sep">◆</span>
        <span class="phi-results-tf">{n_ind} indicator{"s" if n_ind != 1 else ""} · {blend_lbl}</span>
      </div>
      <div class="phi-results-metrics-row">
        <div class="phi-rm-item phi-rm-hero">
          <div class="phi-rm-value" style="color:{pl_color};font-size:2.4rem;">${net_pl:+,.0f}</div>
          <div class="phi-rm-label">Net Profit / Loss</div>
        </div>
        <div class="phi-rm-divider"></div>
        <div class="phi-rm-item">
          <div class="phi-rm-value" style="color:{ret_color};">{tr:+.1%}</div>
          <div class="phi-rm-label">Total Return</div>
        </div>
        <div class="phi-rm-item">
          <div class="phi-rm-value">{cagr:+.1%}</div>
          <div class="phi-rm-label">CAGR</div>
        </div>
        <div class="phi-rm-item">
          <div class="phi-rm-value" style="color:{sh_color};">{sharpe:.2f}</div>
          <div class="phi-rm-label">Sharpe</div>
        </div>
        <div class="phi-rm-item">
          <div class="phi-rm-value" style="color:#ef4444;">{dd:.1%}</div>
          <div class="phi-rm-label">Max Drawdown</div>
        </div>
        <div class="phi-rm-item">
          <div class="phi-rm-value">{acc:.1%}</div>
          <div class="phi-rm-label">Direction Acc</div>
        </div>
        {opt_metrics}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 5 tabs ────────────────────────────────────────────────────────────
    tab_ov, tab_mon, tab_tr, tab_met, tab_rev = st.tabs(
        ["📈 Overview", "📅 Monthly Returns", "🔁 Trades", "📊 Metrics", "🤖 AI Review"]
    )

    # ── Tab 1: Overview ───────────────────────────────────────────────────
    with tab_ov:
        col_chart, col_stats = st.columns([3, 1])
        with col_chart:
            eq_fig = _make_equity_fig(plog, pv, cap)
            if eq_fig:
                st.markdown('<div class="phi-chart-label">EQUITY CURVE</div>', unsafe_allow_html=True)
                _phi_chart(eq_fig, height=340)
            dd_fig = _make_drawdown_fig(pv)
            if dd_fig:
                st.markdown('<div class="phi-chart-label">DRAWDOWN</div>', unsafe_allow_html=True)
                _phi_chart(dd_fig, height=130)
        with col_stats:
            trade_rows = ""
            if n_trades:
                trade_rows = f"""
                <div class="phi-stat-card">
                  <div class="phi-stat-label">Total Trades</div>
                  <div class="phi-stat-value">{n_trades}</div>
                </div>
                <div class="phi-stat-card">
                  <div class="phi-stat-label">Avg Hold (bars)</div>
                  <div class="phi-stat-value">{tstats.get('avg_bars_held', 0):.0f}</div>
                </div>"""
            st.markdown(f"""
            <div class="phi-stat-stack">
              <div class="phi-stat-card">
                <div class="phi-stat-label">Start Capital</div>
                <div class="phi-stat-value">${cap:,.0f}</div>
              </div>
              <div class="phi-stat-card">
                <div class="phi-stat-label">End Capital</div>
                <div class="phi-stat-value" style="color:{pl_color};">${end_cap:,.0f}</div>
              </div>
              {trade_rows}
            </div>
            """, unsafe_allow_html=True)
            blend_fig = _make_blend_fig(indicators, blend_weights)
            if blend_fig:
                st.markdown('<div class="phi-chart-label">BLEND WEIGHTS</div>', unsafe_allow_html=True)
                _phi_chart(blend_fig, height=max(160, 40 + 36 * n_ind))

    # ── Tab 2: Monthly Returns ────────────────────────────────────────────
    with tab_mon:
        if monthly.empty:
            st.info("Need at least 60 days of bars for a monthly breakdown.")
        else:
            hm_fig = _make_monthly_heatmap_fig(monthly)
            if hm_fig:
                st.markdown('<div class="phi-chart-label">MONTHLY RETURNS</div>', unsafe_allow_html=True)
                _phi_chart(hm_fig, height=max(180, 60 + 52 * len(monthly)))
            ann_fig = _make_annual_returns_fig(monthly)
            if ann_fig:
                st.markdown('<div class="phi-chart-label">ANNUAL RETURNS</div>', unsafe_allow_html=True)
                _phi_chart(ann_fig, height=220)

    # ── Tab 3: Trades ─────────────────────────────────────────────────────
    with tab_tr:
        if not plog:
            st.caption("No trade log available.")
        else:
            if tstats:
                lg_win  = tstats.get("largest_win",  0)
                lg_loss = tstats.get("largest_loss", 0)
                av_win  = tstats.get("avg_win",  0)
                av_loss = tstats.get("avg_loss", 0)
                wr_col  = "#22c55e" if win_rate >= 0.5 else "#ef4444"
                pf_col  = "#22c55e" if pf > 1 else "#ef4444"
                st.markdown(f"""
                <div class="phi-trade-stats-grid">
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Total Trades</div>
                    <div class="phi-ts-value">{n_trades}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Win Rate</div>
                    <div class="phi-ts-value" style="color:{wr_col};">{win_rate:.1%}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Profit Factor</div>
                    <div class="phi-ts-value" style="color:{pf_col};">{pf_str}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Avg Hold (bars)</div>
                    <div class="phi-ts-value">{tstats.get("avg_bars_held", 0):.0f}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Avg Win</div>
                    <div class="phi-ts-value" style="color:#22c55e;">${av_win:+,.2f}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Avg Loss</div>
                    <div class="phi-ts-value" style="color:#ef4444;">${av_loss:+,.2f}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Largest Win</div>
                    <div class="phi-ts-value" style="color:#22c55e;">${lg_win:+,.2f}</div>
                  </div>
                  <div class="phi-ts-card">
                    <div class="phi-ts-label">Largest Loss</div>
                    <div class="phi-ts-value" style="color:#ef4444;">${lg_loss:+,.2f}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            df_log = pd.DataFrame(plog)
            csv_bytes = df_log.to_csv(index=False).encode()
            st.download_button(
                label="⬇ Export Trade Log (CSV)",
                data=csv_bytes,
                file_name=f"phi_{symbol}_{start_lbl}_{end_lbl}.csv",
                mime="text/csv",
            )
            st.dataframe(df_log, use_container_width=True, height=320)

    # ── Tab 4: Metrics ────────────────────────────────────────────────────
    with tab_met:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Returns**")
            st.metric("Total Return", f"{tr:+.2%}")
            st.metric("CAGR",         f"{cagr:+.2%}")
        with mc2:
            st.markdown("**Risk**")
            st.metric("Max Drawdown", f"{dd:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        payload: dict = {
            "total_return": tr, "cagr": cagr,
            "max_drawdown": dd, "sharpe": sharpe,
            "direction_accuracy": acc,
        }
        if n_trades:
            payload.update({
                "total_trades": n_trades, "win_rate": win_rate,
                "profit_factor": pf if pf != float("inf") else None,
            })
        if isinstance(results, dict) and results.get("options_snapshot"):
            payload["options_snapshot"] = results["options_snapshot"]
        st.markdown("**Full payload**")
        st.json(payload)

    # ── Tab 5: AI Review ──────────────────────────────────────────────────
    with tab_rev:
        _render_phibot_review(config, results, strat, indicators, blend_method, blend_weights)

    # ── Persist run ───────────────────────────────────────────────────────
    from phi.run_config import RunConfig, RunHistory
    run_cfg = st.session_state.get("validated_run_config")
    if not isinstance(run_cfg, RunConfig):
        _cfg_start = config.get("start", "")
        _cfg_end = config.get("end", "")
        run_cfg = RunConfig(
            symbols=config.get("symbols", ["SPY"]),
            start_date=str(_cfg_start.date()) if hasattr(_cfg_start, "date") else str(_cfg_start),
            end_date=str(_cfg_end.date()) if hasattr(_cfg_end, "date") else str(_cfg_end),
            timeframe=config.get("timeframe", "1D"),
            initial_capital=cap,
            indicators=indicators,
            blend_method=blend_method,
            blend_weights=blend_weights,
            trading_mode=config.get("trading_mode", "equities"),
        )
    hist = RunHistory()
    run_id = hist.create_run(run_cfg)
    hist.save_results(run_id, {
        "total_return": tr, "cagr": cagr, "max_drawdown": dd,
        "sharpe": sharpe, "accuracy": acc, "net_pl": net_pl,
    })
    st.caption(f"Run saved: `{run_id}`")


# ---------------------------------------------------------------------------
# Run & Results Orchestration
# ---------------------------------------------------------------------------
def render_run_and_results(config, indicators, blend_method, blend_weights):
    if not config or not indicators:
        st.info("Complete Steps 1-2 to run a backtest.")
        return

    st.markdown(
        """
        <div style="border-top:1px solid rgba(168,85,247,0.15);
                    margin:2rem 0 1.5rem;"></div>
        <div style="font-size:1.4rem; font-weight:700; color:#e4e4e7;
                    letter-spacing:-0.02em; margin-bottom:0.8rem;">
            Run Backtest
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(_render_section_header("", "RUN BACKTEST", "EXECUTE"), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    c1.selectbox("Primary metric", METRICS, key="primary_metric")
    run_clicked = c2.button("Run Backtest", type="primary", key="run_bt", use_container_width=True)

    phiai_full = st.session_state.get("phiai_full", False)
    trading_mode = config.get("trading_mode", "equities")

    # Show Phibot tweaks-applied confirmation banner (cleared after one render)
    n_applied = st.session_state.pop("phibot_tweaks_applied", None)
    if n_applied:
        st.success(
            f"Phibot applied {n_applied} tweak(s) to your configuration. "
            "Click **Run Backtest** to see the updated results."
        )

        st.markdown("**Evaluation Metric**")
        eval_metric = st.selectbox(
            "Primary Metric",
            ["sharpe", "total_return", "cagr", "sortino", "max_drawdown",
             "direction_accuracy", "profit_factor", "win_rate"],
            format_func=_metric_label,
            key="wb_eval_metric",
        )

        st.markdown("**Run Description** (optional)")
        run_desc = st.text_input("Notes / tag this run", value="", key="wb_run_desc",
                                  placeholder="e.g. SPY daily RSI+MACD blend")

    st.divider()

    # ── Run / Stop buttons ────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
    run_clicked  = btn_col1.button("▶ Run Backtest", type="primary", use_container_width=True)
    stop_clicked = btn_col2.button("■ Stop", use_container_width=True)

    if stop_clicked:
        _set("wb_running", False)
        st.warning("Backtest stopped.")
        return

    if not run_clicked:
        return

    run_payload = {
        "dataset_id": config.get("dataset_id", ""),
        "symbols": config.get("symbols", ["SPY"]),
        "start_date": config.get("start"),
        "end_date": config.get("end"),
        "timeframe": config.get("timeframe", "1D"),
        "vendor": config.get("vendor", "alphavantage"),
        "initial_capital": config.get("initial_capital", 100_000.0),
        "trading_mode": config.get("trading_mode", "equities"),
        "indicators": indicators,
        "blend_method": blend_method,
        "blend_weights": blend_weights,
        "phiai_enabled": bool(st.session_state.get("phiai_full", False)),
        "evaluation_metric": st.session_state.get("wb_eval_metric", "roi"),
        "option_params": {},
    }
    if config.get("trading_mode", "equities") == "options":
        bt_opts = st.session_state.get("bt_options_controls", {})
        symbol = run_payload["symbols"][0] if run_payload["symbols"] else "SPY"
        run_payload["option_params"] = {symbol: bt_opts}

    try:
        from phi.run_config import RunConfig
        validated_cfg = RunConfig.model_validate(run_payload)
    except ValidationError as e:
        st.error("Configuration validation failed. Please fix the following issues:")
        for err in e.errors():
            field = ".".join(str(part) for part in err.get("loc", [])) or "config"
            st.error(f"• {field}: {err.get('msg', 'invalid value')}")
        return

    config["symbols"] = validated_cfg.symbols
    config["start"] = datetime.combine(validated_cfg.start_date, datetime.min.time())
    config["end"] = datetime.combine(validated_cfg.end_date, datetime.min.time())
    config["timeframe"] = validated_cfg.timeframe
    config["vendor"] = validated_cfg.vendor
    config["initial_capital"] = validated_cfg.initial_capital
    config["trading_mode"] = validated_cfg.trading_mode
    st.session_state["validated_run_config"] = validated_cfg

    # Honour any overrides adopted from Phibot review
    _signal_threshold  = float(st.session_state.get("phibot_signal_threshold", 0.15))
    _position_size_pct = float(st.session_state.get("phibot_position_size", 0.95))

    indicators_to_use = dict(indicators)
    phiai_explanation = ""
    workbench_data = st.session_state.get("workbench_dataset") or {}

    # PhiAI optimization
    if phiai_full and workbench_data:
        phiai_progress = st.progress(0, text="PhiAI optimizing indicators and parameters...")
        phiai_result, phiai_exc = [None], [None]
        _sym = config["symbols"][0]
        _ohlcv = workbench_data.get(_sym)

        def run_phiai():
            try:
                if _ohlcv is not None and len(_ohlcv) > 100:
                    from phi.phiai import run_phiai_optimization
                    phiai_result[0] = run_phiai_optimization(
                        _ohlcv, indicators_to_use, max_iter_per_indicator=15
                    )
            except Exception as ex:
                phiai_exc[0] = ex

        th = threading.Thread(target=run_phiai)
        th.start()
        start_t = time.time()
        while th.is_alive():
            time.sleep(0.3)
            pct = min(95, int((time.time() - start_t) * 12))
            phiai_progress.progress(pct / 100, text=f"PhiAI optimizing... {pct}%")
        if phiai_exc[0]:
            st.warning(f"PhiAI skipped: {phiai_exc[0]}")
        elif (r := phiai_result[0]) is not None and isinstance(r, (list, tuple)) and len(r) == 2:
            indicators_to_use, phiai_explanation = r
        phiai_progress.progress(1.0, text="PhiAI complete!")
        time.sleep(0.3)
        phiai_progress.empty()

    # Options backtest
    if trading_mode == "options":
        try:
            opt_progress = st.progress(0, text="Running options backtest...")
            opt_result, opt_exc = [None], [None]

            def run_opt():
                try:
                    dfs = st.session_state.get("workbench_dataset", {})
                    sym = config["symbols"][0]
                    ohlcv = dfs.get(sym)
                    if ohlcv is None or ohlcv.empty:
                        raise ValueError("No data for options backtest. Fetch first.")
                    # pylint: disable=import-outside-toplevel
                    from phi.options import run_options_backtest
                    from phi.run_config import RunConfig

                    cfg = st.session_state.get("validated_run_config")
                    if not isinstance(cfg, RunConfig):
                        raise ValueError("Validated run config missing for options mode.")
                    opt_result[0] = run_options_backtest(cfg, ohlcv)
                except Exception as e:  # pylint: disable=broad-except
                    opt_exc[0] = e

            th_opt = threading.Thread(target=run_opt)
            th_opt.start()
            start_t = time.time()
            while th_opt.is_alive():
                time.sleep(0.2)
                elapsed = time.time() - start_t
                pct = min(95, int(elapsed * 25))
                opt_progress.progress(pct / 100, text=f"Running options backtest... {pct}%")
            if opt_exc[0]:
                opt_progress.empty()
                raise opt_exc[0]  # pylint: disable=raising-bad-type
            results = opt_result[0]
            opt_progress.progress(1.0, text="Complete!")
            time.sleep(0.3)
            opt_progress.empty()
            if results:
                _display_results(config, results, None, indicators_to_use, blend_method, blend_weights)
            else:
                st.error("No results returned.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            st.error(str(e))
        return

    # Equities backtest — direct vectorized path (no Lumibot)
    progress = st.progress(0, text="Running backtest...")

    try:
        sym = config["symbols"][0]
        dfs_bt = st.session_state.get("workbench_dataset", {})
        ohlcv_bt = dfs_bt.get(sym)
        if ohlcv_bt is None or (hasattr(ohlcv_bt, "empty") and ohlcv_bt.empty):
            progress.empty()
            st.error("No dataset loaded. Complete Step 1 (Fetch & Cache Data) first.")
            return

        result_holder, exc_holder = [None], [None]

        def run_bt():
            try:
                from phi.backtest import run_direct_backtest  # pylint: disable=import-outside-toplevel
                result_holder[0] = run_direct_backtest(
                    ohlcv=ohlcv_bt,
                    symbol=sym,
                    indicators=indicators_to_use,
                    blend_weights=blend_weights,
                    blend_method=blend_method,
                    signal_threshold=_signal_threshold,
                    initial_capital=config["initial_capital"],
                    position_size_pct=_position_size_pct,
                )
            except Exception as e:  # pylint: disable=broad-except
                exc_holder[0] = e

        th = threading.Thread(target=run_bt)
        th.start()
        start_t = time.time()
        while th.is_alive():
            time.sleep(0.3)
            pct = min(95, 5 + int((time.time() - start_t) * 8))
            progress.progress(pct / 100, text=f"Running backtest... {pct}%")

        if exc_holder[0]:
            raise exc_holder[0]  # pylint: disable=raising-bad-type
        if result_holder[0] and isinstance(result_holder[0], (list, tuple)) and len(result_holder[0]) == 2:
            results, strat = result_holder[0]
            progress.progress(1.0, text="Complete!")
            time.sleep(0.5)
            progress.empty()
            sc = _compute_accuracy(strat) if hasattr(strat, "prediction_log") else {}
            _display_results(config, results, strat, indicators_to_use, blend_method, blend_weights, sc)
        else:
            engine = BacktestEngine(backtest_cfg)
            result_obj = engine.run(ohlcv, blended_signal, progress_callback=_bt_progress)

        for line in result_obj.run_log if hasattr(result_obj, "run_log") else []:
            _log(line)

        # ── 6. Compute metrics ────────────────────────────────────────────────
        _status("Step 6/6 — Computing metrics...", 0.85)
        metrics = result_obj.metrics

        end_cap   = result_obj.end_capital
        start_cap = cfg.initial_capital
        net_pnl   = end_cap - start_cap

        _log(f"Trades: {metrics.get('n_trades', 0):,}  |  "
             f"Win rate: {metrics.get('win_rate', 0):.1%}  |  "
             f"Sharpe: {metrics.get('sharpe', 0):.2f}")
        _log(f"End capital: ${end_cap:,.2f}  (net P/L: ${net_pnl:+,.2f})")

        # ── Save run ─────────────────────────────────────────────────────────
        _status("Saving run to history...", 0.95)
        run_data = {
            "metrics":     metrics,
            "end_capital": end_cap,
            "start_capital": start_cap,
            "equity_curve": result_obj.equity_curve,
            "signals":      blended_signal,
        }
        try:
            save_run(cfg, run_data, result_obj.trades)
            _log(f"Run saved: runs/{cfg.run_id}/", "info")
        except Exception as e:
            _log(f"Save failed: {e}", "warn")

        _set("wb_last_result", {
            "run_id":      cfg.run_id,
            "config":      cfg,
            "result":      result_obj,
            "signals_df":  signals_df,
            "blended":     blended_signal,
            "metrics":     metrics,
            "end_capital": end_cap,
        })

        _status("✓ Backtest complete!", 1.0)
        time.sleep(0.3)
        progress_bar.empty()
        status_area.empty()
        log_area.empty()
        st.rerun()

    except Exception as e:  # pylint: disable=broad-exception-caught
        progress.empty()
        st.error(str(e))
        st.exception(e)


def _wrap_options_result(opt_result):
    """Wrap OptionsBacktestResult into a duck-typed object matching BacktestResult interface."""
    class _Wrapper:
        pass
    w = _Wrapper()
    w.equity_curve  = opt_result.equity_curve
    w.trades        = opt_result.trades
    w.metrics       = opt_result.metrics
    w.start_capital = opt_result.start_capital
    w.end_capital   = opt_result.end_capital
    w.signals       = pd.Series(dtype=float)
    w.positions     = pd.Series(dtype=float)
    w.run_log       = []
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_results():
    last = _ss("wb_last_result")
    if last is None:
        return

    result     = last["result"]
    metrics    = last["metrics"]
    cfg        = last["config"]
    signals_df = last.get("signals_df", pd.DataFrame())
    blended    = last.get("blended", pd.Series(dtype=float))
    run_id     = last["run_id"]
    end_cap    = last["end_capital"]
    start_cap  = cfg.initial_capital
    net_pnl    = end_cap - start_cap
    net_pnl_pct = net_pnl / (start_cap + 1e-10)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Results")

    # Eval metric value
    em   = cfg.evaluation_metric
    em_v = metrics.get(em, metrics.get("total_return", 0))
    em_lbl = _metric_label(em)
    if em in ("total_return", "cagr", "win_rate", "direction_accuracy", "max_drawdown"):
        em_str = f"{float(em_v):.2%}" if em_v is not None else "—"
    else:
        em_str = f"{float(em_v):.3f}" if em_v is not None else "—"

    st.markdown(
        result_ribbon_html(start_cap, end_cap, net_pnl, net_pnl_pct, em_lbl, em_str),
        unsafe_allow_html=True,
    )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Summary", "Equity Curve", "Trades", "Metrics", "Diagnostics"])

    # ── Summary ───────────────────────────────────────────────────────────────
    with tabs[0]:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Total Return",  _fmt_pct(metrics.get("total_return")))
        rc2.metric("CAGR",          _fmt_pct(metrics.get("cagr")))
        rc3.metric("Sharpe",        _fmt_num(metrics.get("sharpe"), 2))
        rc4.metric("Max Drawdown",  _fmt_pct(metrics.get("max_drawdown")))

        rc5, rc6, rc7, rc8 = st.columns(4)
        rc5.metric("Win Rate",      _fmt_pct(metrics.get("win_rate")))
        rc6.metric("Profit Factor", _fmt_num(metrics.get("profit_factor"), 2))
        rc7.metric("Trades",        f"{metrics.get('n_trades', 0):,}")
        rc8.metric("Dir. Accuracy", _fmt_pct(metrics.get("direction_accuracy")))

        st.markdown(f"**Run ID:** `{run_id}`  |  **Symbol:** {cfg.symbol}  |  "
                    f"**Timeframe:** {cfg.timeframe}  |  **Mode:** {cfg.trading_mode.title()}")
        st.caption(
            f"Indicators: {', '.join(i.display_name for i in cfg.indicators)}  |  "
            f"Blend: {cfg.blend_mode}  |  "
            f"PhiAI: {'on' if cfg.phiai_enabled else 'off'}"
        )

        ec1, ec2 = st.columns(2)
        with ec1:
            if st.button("Export Config (JSON)", key="exp_cfg", use_container_width=True):
                st.download_button(
                    "Download config.json",
                    data=cfg.to_json(),
                    file_name=f"run_{run_id}_config.json",
                    mime="application/json",
                )
        with ec2:
            if not result.trades.empty and st.button("Export Trades (CSV)",
                                                      key="exp_trades", use_container_width=True):
                st.download_button(
                    "Download trades.csv",
                    data=result.trades.to_csv(index=False),
                    file_name=f"run_{run_id}_trades.csv",
                    mime="text/csv",
                )

    # ── Equity Curve ──────────────────────────────────────────────────────────
    with tabs[1]:
        eq = result.equity_curve.to_frame("Portfolio Value ($)")
        if not blended.empty:
            # Normalize blended to overlay on equity chart
            pass
        st.line_chart(eq, use_container_width=True)

        # Drawdown chart
        if len(result.equity_curve) > 1:
            dd_series = (result.equity_curve - result.equity_curve.cummax()) / \
                        result.equity_curve.cummax()
            st.markdown("**Drawdown**")
            st.area_chart(dd_series.to_frame("Drawdown"), use_container_width=True)

    # ── Trades ────────────────────────────────────────────────────────────────
    with tabs[2]:
        if result.trades.empty:
            st.warning("No trades executed. Try lowering the signal threshold.")
        else:
            st.dataframe(result.trades, use_container_width=True, hide_index=True)

            if "pnl" in result.trades.columns:
                t1, t2 = st.columns(2)
                with t1:
                    pnl_hist = result.trades["pnl"]
                    pnl_df = pnl_hist.to_frame("Trade P&L ($)")
                    st.bar_chart(pnl_df, use_container_width=True)
                with t2:
                    cum_pnl = result.trades["pnl"].cumsum()
                    st.line_chart(cum_pnl.to_frame("Cumulative P&L ($)"),
                                  use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    with tabs[3]:
        metric_groups = [
            ("Returns", ["total_return", "cagr", "net_pnl", "net_pnl_pct"]),
            ("Risk",    ["max_drawdown", "volatility_annual", "sharpe", "sortino", "calmar"]),
            ("Trades",  ["n_trades", "win_rate", "profit_factor", "avg_win",
                         "avg_loss", "largest_win", "largest_loss", "avg_hold_bars"]),
            ("Accuracy", ["direction_accuracy"]),
            ("Capital", ["initial_capital", "end_capital"]),
        ]
        for group_name, keys in metric_groups:
            st.markdown(f"**{group_name}**")
            cols = st.columns(len(keys))
            for col, k in zip(cols, keys):
                v = metrics.get(k)
                if v is None:
                    col.metric(k.replace("_", " ").title(), "—")
                elif k in ("total_return", "cagr", "win_rate", "max_drawdown",
                           "volatility_annual", "direction_accuracy", "net_pnl_pct"):
                    col.metric(k.replace("_", " ").title(), _fmt_pct(v))
                elif k in ("initial_capital", "end_capital", "net_pnl",
                           "avg_win", "avg_loss", "largest_win", "largest_loss"):
                    col.metric(k.replace("_", " ").title(), _fmt_usd(v))
                elif k in ("n_trades",):
                    col.metric(k.replace("_", " ").title(), f"{int(v):,}")
                else:
                    col.metric(k.replace("_", " ").title(), _fmt_num(v, 3))
            st.markdown("")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    with tabs[4]:
        if not signals_df.empty:
            st.markdown("**Individual Indicator Signals**")
            st.line_chart(signals_df.tail(300), use_container_width=True)

        if not blended.empty:
            st.markdown("**Blended Signal**")
            st.line_chart(blended.tail(300).to_frame("Blended Signal"),
                          use_container_width=True)

        if hasattr(result, "positions") and not result.positions.empty:
            st.markdown("**Position Series (+1=long, -1=short, 0=flat)**")
            st.line_chart(result.positions.tail(300).to_frame("Position"),
                          use_container_width=True)

        # PhiAI explanation
        phiai_res = _ss("wb_phiai_result")
        if phiai_res and phiai_res.get("explanation"):
            with st.expander("PhiAI Explanation", expanded=True):
                st.code(phiai_res["explanation"], language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN HISTORY TAB
# ═══════════════════════════════════════════════════════════════════════════════

def render_run_history():
    st.markdown("## Run History")

    runs = list_runs(limit=50)

    if not runs:
        st.info("No runs stored yet. Complete a backtest to see history.")
        return

    # Summary table
    display_cols = ["run_id", "created_at", "symbol", "timeframe", "trading_mode",
                    "indicators", "blend_mode", "phiai", "initial_capital",
                    "end_capital", "total_return", "sharpe", "max_drawdown",
                    "win_rate", "n_trades"]
    df_runs = pd.DataFrame(runs)
    avail_cols = [c for c in display_cols if c in df_runs.columns]
    st.dataframe(df_runs[avail_cols], use_container_width=True, hide_index=True)

    # Compare multiple runs
    run_ids = [r["run_id"] for r in runs]
    selected_ids = st.multiselect("Select runs to compare", run_ids, default=run_ids[:min(3, len(run_ids))],
                                   key="wb_compare_ids")

    if selected_ids and st.button("Compare Selected Runs", use_container_width=True):
        cmp_df = compare_runs(selected_ids)
        if not cmp_df.empty:
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)

            # Equity curve comparison
            st.markdown("**Equity Curves**")
            eq_data: Dict[str, pd.Series] = {}
            for rid in selected_ids:
                run = load_run(rid)
                if run and run.get("results"):
                    eq = run["results"].get("equity_curve")
                    if isinstance(eq, (pd.Series, pd.DataFrame)):
                        lbl = f"{run['results'].get('symbol', rid)} ({rid})"
                        eq_data[lbl] = eq if isinstance(eq, pd.Series) else eq.iloc[:, 0]
                    elif isinstance(eq, dict):
                        # Stored as dict
                        vals = list(eq.values())
                        if vals and isinstance(vals[0], list):
                            eq_data[rid] = pd.Series(vals[0])
            if eq_data:
                eq_df = pd.DataFrame(eq_data)
                st.line_chart(eq_df, use_container_width=True)

    # Delete
    with st.expander("Manage Runs", expanded=False):
        del_id = st.selectbox("Delete run", ["— select —"] + run_ids, key="wb_del_run")
        if del_id != "— select —":
            if st.button(f"Delete run {del_id}", key="wb_del_btn"):
                delete_run(del_id)
                st.success(f"Deleted run {del_id}")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE MANAGER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def render_cache_manager():
    st.markdown("## Cache Manager")

    cached = list_cached_datasets()

    if not cached:
        st.info("No cached datasets. Use Step 1 to fetch and cache data.")
        return

    display = []
    for c in cached:
        display.append({
            "dataset_id": c.get("dataset_id", ""),
            "symbol":     c.get("symbol", ""),
            "timeframe":  c.get("timeframe", ""),
            "vendor":     c.get("vendor", ""),
            "start":      c.get("start", ""),
            "end":        c.get("end", ""),
            "rows":       c.get("rows", 0),
            "cached_at":  c.get("cached_at", "")[:19],
        })

    df_cache = pd.DataFrame(display)
    st.dataframe(df_cache, use_container_width=True, hide_index=True)

    if st.button("Clear All Cache", key="wb_clear_cache"):
        n = clear_all_cache()
        st.success(f"Cleared {n} cached datasets.")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    render_banner()

    # Top-level navigation tabs
    tab_wb, tab_hist, tab_cache = st.tabs([
        "⚗ Workbench",
        "📋 Run History",
        "💾 Cache Manager",
    ])

    with tab_wb:
        # ── Step 1 ────────────────────────────────────────────────────────────
        dataset = render_step1()

        if dataset is None:
            st.info("Complete Step 1 to continue.")
            # Show last result even if dataset not loaded
            render_results()
            return

        st.divider()

        # ── Step 2 ────────────────────────────────────────────────────────────
        indicators = render_step2(dataset)

        st.divider()

        # ── Step 3 (if 2+ indicators) ─────────────────────────────────────────
        active_inds = [i for i in indicators if i.get("enabled", True)]
        if len(active_inds) >= 2:
            blend_cfg = render_step3(indicators)
            st.divider()
        else:
            blend_cfg = {"mode": "weighted_sum", "weights": {}, "regime_weights": {}}

        # ── Step 4 ────────────────────────────────────────────────────────────
        phiai_cfg = render_step4(indicators)

        st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# AI Agents (Ollama)
# ─────────────────────────────────────────────────────────────────────────────
def render_ai_agents():
    """Ollama-powered free AI agents for regime/strategy Q&A."""
    st.markdown("### AI Agents (Ollama)")
    st.caption(
        "Free local models via [Ollama](https://ollama.com). "
        "Install: [ollama.com/download](https://ollama.com/download) · "
        "Then: `ollama pull llama3.2` or `ollama pull 0xroyce/plutus`"
    )

    host = st.text_input(
        "Ollama Host",
        value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        key="ollama_host",
        help="Default: http://localhost:11434",
    )

    col_check, col_list = st.columns(2)
    with col_check:
        if st.button("Check connection"):
            # pylint: disable=import-outside-toplevel
            from phi.agents import check_ollama_ready
            ok = check_ollama_ready(host)
            if ok:
                st.success("Ollama is running")
            else:
                st.error(
                    "Ollama not reachable. Install from ollama.com and run it."
                )
    with col_list:
        if st.button("List models"):
            # pylint: disable=import-outside-toplevel
            from phi.agents import list_ollama_models
            models = list_ollama_models(host)
            if models:
                names = [
                    m.get("name", "").split(":")[0]
                    for m in models if m.get("name")
                ]
                st.session_state["ollama_models"] = list(dict.fromkeys(names))
                st.success(
                    f"Found {len(st.session_state['ollama_models'])} model(s)"
                )
            else:
                msg = (
                    "No models or Ollama not running. "
                    "Run: ollama pull llama3.2"
                )
                st.warning(msg)

    m_def = ["llama3.2", "0xroyce/plutus"]
    model_list = st.session_state.get("ollama_models", m_def)
    model = st.selectbox(
        "Model",
        options=model_list,
        index=0 if model_list else 0,
        key="ollama_model",
        help="Pull with: ollama pull <model>",
    )

    prompt = st.text_area(
        "Ask AI (regime, strategy, indicators, etc.)",
        placeholder=(
            "e.g. What does RSI oversold mean? "
            "When to use regime-weighted blending?"
        ),
        key="ollama_prompt",
        height=80,
    )
    if st.button("Send", key="ollama_send") and prompt:
        with st.spinner("Thinking..."):
            try:
                from phi.agents import OllamaAgent
                agent = OllamaAgent(model=model, host=host)
                reply = agent.chat(
                    prompt,
                    system=(
                        "You are a quantitative trading assistant. "
                        "Be concise."
                    ),
                )
                st.markdown("**Reply:**")
                st.markdown(reply)
            except Exception as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Run History & Cache
# ---------------------------------------------------------------------------
def render_run_history():
    """Render validated history from previous backtest runs."""
    # pylint: disable=import-outside-toplevel
    from phi.run_config import RunConfig, RunHistory
    hist = RunHistory()
    runs = hist.list_runs()
    if not runs:
        st.caption("No runs yet.")
        return

    st.markdown("### Run History")
    for r in runs[:10]:
        run_id = r["run_id"]
        run_dir = hist.root / run_id
        with st.expander(run_id):
            try:
                cfg = RunConfig.load(run_dir)
                st.json({"config": cfg.model_dump(), "results": r.get("results", {})})
            except ValidationError as exc:
                st.error(f"Failed to validate saved config for {run_id}.")
                for err in exc.errors():
                    field = ".".join(str(part) for part in err.get("loc", [])) or "config"
                    st.error(f"• {field}: {err.get('msg', 'invalid value')}")
                st.json({"results": r.get("results", {})})

        render_results()

# ─────────────────────────────────────────────────────────────────────────────
# Cache Manager
# ─────────────────────────────────────────────────────────────────────────────
def render_cache_manager():
    from phi.data import list_cached_datasets
    datasets = list_cached_datasets()
    st.markdown("### Cache Manager")
    if not datasets:
        st.caption("No cached datasets.")
        return
    st.dataframe(pd.DataFrame(datasets), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Phi-nance Live Workbench", page_icon="📊", layout="wide")

    _inject_css()
    _sidebar()
    _render_update_banner()

    # ── Hero header ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg,
                rgba(168,85,247,0.1) 0%,
                rgba(124,58,237,0.06) 50%,
                rgba(249,115,22,0.05) 100%);
            border: 1px solid rgba(168,85,247,0.2);
            border-radius: 16px;
            padding: 1.8rem 2.2rem;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position:absolute; top:-60%; left:-10%; width:50%; height:220%;
                background: radial-gradient(ellipse,
                    rgba(168,85,247,0.06) 0%, transparent 70%);
                pointer-events:none;
            "></div>
            <div style="display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;">
                <div style="font-size:2.5rem; line-height:1;">📊</div>
                <div style="flex:1; min-width:200px;">
                    <div style="
                        font-size:1.7rem; font-weight:800;
                        background:linear-gradient(135deg,#a855f7 0%,#c084fc 45%,#f97316 100%);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; letter-spacing:-0.03em; line-height:1.1;
                    ">Phi-nance Live Workbench</div>
                    <div style="color:#71717a; font-size:0.9rem; margin-top:0.3rem;">
                        Regime-aware quant research &nbsp;&bull;&nbsp;
                        Fetch &rarr; Select &rarr; Blend &rarr; PhiAI &rarr; Run
                    </div>
                </div>
                <div style="display:flex; gap:0.6rem; flex-wrap:wrap; align-items:center;">
                    <span style="
                        background:rgba(168,85,247,0.12); color:#c084fc;
                        border:1px solid rgba(168,85,247,0.25);
                        border-radius:20px; padding:0.25rem 0.75rem;
                        font-size:0.75rem; font-weight:600; letter-spacing:0.04em;
                    ">DARK THEME</span>
                    <span style="
                        background:rgba(249,115,22,0.1); color:#fb923c;
                        border:1px solid rgba(249,115,22,0.25);
                        border-radius:20px; padding:0.25rem 0.75rem;
                        font-size:0.75rem; font-weight:600; letter-spacing:0.04em;
                    ">CACHED</span>
                    <span style="
                        background:rgba(34,197,94,0.08); color:#4ade80;
                        border:1px solid rgba(34,197,94,0.2);
                        border-radius:20px; padding:0.25rem 0.75rem;
                        font-size:0.75rem; font-weight:600; letter-spacing:0.04em;
                    ">REPRODUCIBLE</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Fully Automated (one-click) ─────────────────────────────────────────
    with st.expander("⚡ Run Fully Automated — one click to results", expanded=True):
        st.caption(
            "Fetch data → AI picks indicators → tune params → run backtest. "
            "Uses Ollama when available for smarter selection."
        )
        fa_col1, fa_col2, fa_col3 = st.columns(3)
        with fa_col1:
            fa_sym_q = st.text_input("Symbol", value="SPY", key="fa_sym_q")
            fa_start_q = st.date_input("Start", value=date(2020, 1, 1), key="fa_start_q")
        with fa_col2:
            fa_end_q = st.date_input("End", value=date(2024, 12, 31), key="fa_end_q")
            fa_cap_q = st.number_input("Capital ($)", value=100_000, min_value=1000, key="fa_cap_q")
        with fa_col3:
            fa_ollama_q = st.checkbox("Use Ollama", value=True, key="fa_ollama_q")
            fa_host_q = st.text_input("Host", value="http://localhost:11434", key="fa_host_q")
        if st.button("⚡ Run", type="primary", key="fa_run_q", use_container_width=True):
            _run_fully_automated(
                fa_sym_q or "SPY", str(fa_start_q), str(fa_end_q),
                fa_cap_q, fa_ollama_q, fa_host_q
            )
    _inject_css()

    # Device detection (JS already injected by _inject_css)
    dev = detect_device(skip_js=True)

    # Premium header with animated gradient
    if dev.is_phone:
        st.markdown("""
        <div class="phi-hero" style="padding:1.2rem 1rem;">
            <div class="phi-hero-badge">WORKBENCH</div>
            <div class="phi-hero-title" style="font-size:1.6rem;">Phi-nance</div>
            <div class="phi-hero-subtitle" style="font-size:0.8rem;">
                Fetch &rarr; Select &rarr; Blend &rarr; Run
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="phi-hero" style="padding:2rem 2rem 1.5rem;">
            <div class="phi-hero-badge">LIVE BACKTEST WORKBENCH</div>
            <div class="phi-hero-title" style="font-size:2.4rem;">Phi-nance Workbench</div>
            <div class="phi-hero-subtitle" style="font-size:0.92rem;">
                Fetch &rarr; Select &rarr; Blend &rarr; PhiAI &rarr; Run &mdash; Reproducible. Cached. Premium.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Step progress indicator
    has_data = bool(st.session_state.get("workbench_dataset"))
    has_indicators = bool(st.session_state.get("workbench_indicators"))
    steps = [
        ("Data", "completed" if has_data else "active"),
        ("Indicators", "completed" if has_indicators else ("active" if has_data else "pending")),
        ("Blend", "active" if has_indicators and has_data else "pending"),
        ("PhiAI", "pending"),
        ("Run", "pending"),
    ]
    st.markdown(_render_step_indicator(steps), unsafe_allow_html=True)

    st.markdown('<div class="phi-gradient-bar"></div>', unsafe_allow_html=True)

    # Fully Automated section
    with st.expander("Run Fully Automated", expanded=False):
        st.caption("One click: fetch data -> AI picks indicators -> tune params -> backtest.")
        fa1, fa2, fa3 = st.columns(3)
        with fa1:
            fa_sym = st.text_input("Symbol", value="SPY", key="fa_sym")
            fa_start = st.date_input("Start", value=date(2020, 1, 1), key="fa_start")
        with fa2:
            fa_end = st.date_input("End", value=date(2024, 12, 31), key="fa_end")
            fa_cap = st.number_input("Capital ($)", value=100_000, min_value=1000, key="fa_cap")
        with fa3:
            fa_ollama = st.checkbox("Use Ollama", value=True, key="fa_ollama")
            fa_host = st.text_input("Host", value="http://localhost:11434", key="fa_host")

        if st.button("Run Fully Automated", type="primary", key="fa_run", use_container_width=True):
            _run_fully_automated(fa_sym or "SPY", str(fa_start), str(fa_end), fa_cap, fa_ollama, fa_host)

    # ── Divider ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            display:flex; align-items:center; gap:1rem;
            margin:1.5rem 0 0.5rem; color:#52525b;
        ">
            <div style="flex:1; border-top:1px solid rgba(168,85,247,0.12);"></div>
            <div style="font-size:0.8rem; font-weight:600; letter-spacing:0.06em;
                        text-transform:uppercase; white-space:nowrap;">
                Or configure step-by-step
            </div>
            <div style="flex:1; border-top:1px solid rgba(168,85,247,0.12);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Step-by-step workflow
    config = render_dataset_builder()
    indicators = render_indicator_selection()
    blend_method, blend_weights = "weighted_sum", {}
    if len(indicators) >= 2:
        blend_method, blend_weights = render_blending(indicators)
    render_phiai()
    if config:
        render_backtest_controls(config)
    render_run_and_results(config, indicators, blend_method, blend_weights)

    st.markdown(
        """
        <div style="border-top:1px solid rgba(168,85,247,0.12);
                    margin:2.5rem 0 1rem;"></div>
        """,
        unsafe_allow_html=True,
    )
    tab_opts, tab_hist, tab_cache, tab_agents = st.tabs(
        ["Options", "Run History", "Cache Manager", "AI Agents"]
    )
    with tab_opts:
        render_options_workbench()
    with tab_hist:
        render_run_history()

    with tab_cache:
        render_cache_manager()


def render_options_workbench():
    """Phase 3 options workbench with dynamic strategy controls and pricing preview."""
    from phi.options.iv_surface import IVSurface
    from phi.options.strategies import (
        ButterflySpread,
        CalendarSpread,
        Collar,
        CoveredCall,
        DiagonalSpread,
        ProtectivePut,
        SingleLeg,
        Straddle,
        Strangle,
        VerticalSpread,
    )

    st.markdown("### Options")
    sym_default = (st.session_state.get("workbench_config", {}) or {}).get("symbols", ["SPY"])[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Underlying Symbol", value=sym_default, key="opt_symbol_phase3").upper()
    with c2:
        expiry_mode = st.selectbox("Expiry selection", ["Nearest Monthly", "Next Weekly", "Specific Date", "Days to Expiry"], key="opt_expiry_mode")
    with c3:
        qty = int(st.number_input("Contracts", min_value=1, value=1, key="opt_qty_phase3"))

    strategy_map = {
        "Single Leg": SingleLeg,
        "Vertical Spread": VerticalSpread,
        "Straddle": Straddle,
        "Strangle": Strangle,
        "Butterfly Spread": ButterflySpread,
        "Calendar Spread": CalendarSpread,
        "Diagonal Spread": DiagonalSpread,
        "Covered Call": CoveredCall,
        "Protective Put": ProtectivePut,
        "Collar": Collar,
    }
    strategy_name = st.selectbox("Strategy", list(strategy_map.keys()), key="opt_strategy_phase3")

    if expiry_mode == "Specific Date":
        expiry_dt = st.date_input("Expiry Date", value=date.today(), key="opt_expiry_date")
        expiry_years = max((pd.Timestamp(expiry_dt) - pd.Timestamp(date.today())).days / 365.0, 7 / 365)
    elif expiry_mode == "Days to Expiry":
        expiry_days = int(st.slider("Expiry (days)", 7, 365, 30, key="opt_expiry_days"))
        expiry_years = expiry_days / 365.0
    elif expiry_mode == "Next Weekly":
        expiry_years = 7 / 365.0
    else:
        expiry_years = 30 / 365.0

    spot = float(st.number_input("Spot", min_value=1.0, value=100.0, step=1.0, key="opt_spot_phase3"))
    rate = float(st.number_input("Risk-free rate", min_value=0.0, max_value=0.2, value=0.01, step=0.001, key="opt_rate_phase3"))
    base_strike = float(st.number_input("Base Strike", min_value=1.0, value=spot, step=1.0, key="opt_strike_phase3"))

    lower = float(st.number_input("Lower Strike", min_value=1.0, value=max(1.0, base_strike * 0.95), step=1.0, key="opt_lower_strike"))
    upper = float(st.number_input("Upper Strike", min_value=1.0, value=base_strike * 1.05, step=1.0, key="opt_upper_strike"))
    middle = float(st.number_input("Middle Strike", min_value=1.0, value=base_strike, step=1.0, key="opt_middle_strike"))
    option_type = st.selectbox("Option Type", ["call", "put"], key="opt_type_phase3")
    action = st.selectbox("Action", ["buy", "sell"], key="opt_action_phase3")

    cls = strategy_map[strategy_name]
    if strategy_name == "Single Leg":
        strategy = cls(option_type=option_type, action=action, strike=base_strike, expiry=expiry_years, quantity=qty)
    elif strategy_name == "Vertical Spread":
        strategy = cls(option_type=option_type, lower_strike=min(lower, upper), upper_strike=max(lower, upper), expiry=expiry_years, quantity=qty, bullish=True)
    elif strategy_name == "Straddle":
        strategy = cls(strike=base_strike, expiry=expiry_years, quantity=qty)
    elif strategy_name == "Strangle":
        strategy = cls(lower_strike=min(lower, upper), upper_strike=max(lower, upper), expiry=expiry_years, quantity=qty)
    elif strategy_name == "Butterfly Spread":
        strategy = cls(option_type=option_type, lower_strike=min(lower, middle), middle_strike=middle, upper_strike=max(upper, middle), expiry=expiry_years, quantity=qty)
    elif strategy_name == "Calendar Spread":
        strategy = cls(option_type=option_type, strike=base_strike, near_expiry=max(expiry_years * 0.6, 7 / 365), far_expiry=expiry_years, quantity=qty)
    elif strategy_name == "Diagonal Spread":
        strategy = cls(option_type=option_type, short_strike=base_strike, long_strike=upper, near_expiry=max(expiry_years * 0.6, 7 / 365), far_expiry=expiry_years, quantity=qty)
    elif strategy_name == "Covered Call":
        strategy = cls(strike=upper, expiry=expiry_years, quantity=qty)
    elif strategy_name == "Protective Put":
        strategy = cls(strike=lower, expiry=expiry_years, quantity=qty)
    else:
        strategy = cls(put_strike=lower, call_strike=upper, expiry=expiry_years, quantity=qty)

    iv_df = pd.DataFrame(
        {"strike": [spot * 0.8, spot * 0.9, spot, spot * 1.1, spot * 1.2], "expiry": [expiry_years] * 5, "iv": [0.28, 0.24, 0.22, 0.24, 0.27]}
    )
    iv_surface = IVSurface(iv_df)
    premium = strategy.net_premium(spot, rate, iv_surface)
    gs = strategy.greeks(spot, rate, iv_surface)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Net Premium", f"${premium:,.2f}")
    k2.metric("Delta", f"{gs['delta']:.2f}")
    k3.metric("Gamma", f"{gs['gamma']:.4f}")
    k4.metric("Theta", f"{gs['theta']:.2f}")
    k5.metric("Vega", f"{gs['vega']:.2f}")
    k6.metric("Rho", f"{gs['rho']:.2f}")

    price_grid = np.linspace(spot * 0.6, spot * 1.4, 100)
    payoff = np.zeros_like(price_grid)
    for leg in strategy.legs():
        if leg.option_type == "call":
            leg_intr = np.maximum(price_grid - leg.strike, 0.0)
        else:
            leg_intr = np.maximum(leg.strike - price_grid, 0.0)
        leg_intr *= (1 if leg.action == "buy" else -1) * leg.quantity * 100
        payoff += leg_intr
    payoff -= premium

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_grid, y=payoff, mode="lines", name="Payoff @ Expiry"))
    fig.add_hline(y=0, line_dash="dash", line_color="#999")
    fig.update_layout(title=f"{strategy_name} payoff diagram", xaxis_title="Underlying at expiry", yaxis_title="P&L")
    _phi_chart(fig, height=350)

    controls = {
        "symbol": symbol,
        "strategy": strategy_name,
        "quantity": qty,
        "expiry_years": expiry_years,
        "position_pct": float(st.slider("Position size (% of capital)", 1, 100, 10, key="opt_pos_pct_phase3")) / 100.0,
    }
    st.session_state["bt_options_controls"] = controls
    st.caption("Options controls saved and will be included in options backtests.")


if __name__ == "__main__":
    main()

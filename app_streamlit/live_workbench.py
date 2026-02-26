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

import os, sys, time, threading, importlib
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("IS_BACKTESTING", "True")

from app_streamlit.device_detect import detect_device, get_device, inject_responsive_meta, _JS_DETECT

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
}

BLEND_METHODS = ["Weighted Sum", "Regime-Weighted", "Voting", "PhiAI Chooses"]
METRICS = ["ROI", "CAGR", "Sharpe", "Max Drawdown", "Direction Accuracy", "Profit Factor"]
EXIT_STRATEGIES = ["Signal exit", "SL/TP", "Trailing stop", "Time exit"]
POSITION_SIZING = ["Fixed %", "Fixed shares"]


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
    dev = get_device()
    st.markdown(_render_section_header("", "STEP 1 -- DATASET BUILDER", "DATA"), unsafe_allow_html=True)

    if dev.is_phone:
        trading_mode = st.selectbox("Trading Mode", ["Equities", "Options"], key="ds_mode")
        symbols_raw = st.text_input("Symbol(s)", value="SPY", key="ds_symbols",
                                     help="Comma-separated: SPY, QQQ, AAPL")
        start_d = st.date_input("Start", value=date(2020, 1, 1), key="ds_start")
        end_d = st.date_input("End", value=date(2024, 12, 31), key="ds_end")
        timeframe = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m", "5m", "1m"], key="ds_tf")
        vendor = st.selectbox("Data Vendor", ["Alpha Vantage", "yfinance", "Binance Public"], key="ds_vendor")
        initial_capital = st.number_input("Initial Capital ($)", value=100_000, min_value=1_000,
                                           step=10_000, key="ds_cap")
    else:
        col_mode, col_sym, col_range = st.columns([1, 2, 2])
        with col_mode:
            trading_mode = st.selectbox("Trading Mode", ["Equities", "Options"], key="ds_mode")
        with col_sym:
            symbols_raw = st.text_input("Symbol(s)", value="SPY", key="ds_symbols",
                                         help="Comma-separated: SPY, QQQ, AAPL")
        with col_range:
            start_d = st.date_input("Start", value=date(2020, 1, 1), key="ds_start")
            end_d = st.date_input("End", value=date(2024, 12, 31), key="ds_end")

        col_tf, col_vendor, col_cap = st.columns(3)
        with col_tf:
            timeframe = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m", "5m", "1m"], key="ds_tf")
        with col_vendor:
            vendor = st.selectbox("Data Vendor", ["Alpha Vantage", "yfinance", "Binance Public"], key="ds_vendor")
        with col_cap:
            initial_capital = st.number_input("Initial Capital ($)", value=100_000, min_value=1_000,
                                               step=10_000, key="ds_cap")

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
        st.warning("Enter at least one symbol.")
        return None

    dfs = {}
    if fetch_clicked or use_cached:
        from phi.data import fetch_and_cache, get_cached_dataset
        with st.status("Loading data...", expanded=True) as s:
            for sym in symbols:
                try:
                    df = fetch_and_cache(vendor_key, sym, timeframe, str(start_d), str(end_d)) if fetch_clicked \
                         else get_cached_dataset(vendor_key, sym, timeframe, str(start_d), str(end_d))
                    if df is not None and not df.empty:
                        dfs[sym] = df
                except Exception as e:
                    st.error(f"{sym}: {e}")
            if dfs:
                st.session_state["workbench_dataset"] = dfs
                st.session_state["workbench_config"] = {
                    "trading_mode": trading_mode.lower(), "symbols": symbols,
                    "start": datetime.combine(start_d, datetime.min.time()),
                    "end": datetime.combine(end_d, datetime.min.time()),
                    "timeframe": timeframe, "vendor": vendor_key,
                    "initial_capital": float(initial_capital), "benchmark": symbols[0],
                }
                s.update(label=f"Loaded {sum(len(d) for d in dfs.values()):,} bars", state="complete")
            else:
                s.update(label="No data", state="error")

    if st.session_state.get("workbench_dataset"):
        dfs = st.session_state["workbench_dataset"]
        cfg = st.session_state.get("workbench_config", {})

        kpis = [
            ("Symbols", ", ".join(dfs.keys()), "", "neutral"),
            ("Total Bars", f"{sum(len(d) for d in dfs.values()):,}", "", "neutral"),
            ("Capital", f"${cfg.get('initial_capital', 0):,.0f}", "", "neutral"),
            ("Mode", cfg.get("trading_mode", "equities").title(), "", "neutral"),
        ]
        st.markdown(_render_kpi_row(kpis), unsafe_allow_html=True)

        for sym, df in list(dfs.items())[:3]:
            with st.expander(f"{sym} -- {len(df):,} rows"):
                # Mini chart
                if "close" in df.columns and len(df) > 10:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["close"], mode="lines",
                        line=dict(color="#a855f7", width=1.8, shape="spline", smoothing=0.6),
                        fill="tozeroy", fillcolor="rgba(168,85,247,0.04)",
                        hovertemplate="$%{y:.2f}<extra></extra>",
                    ))
                    fig.update_layout(
                        title=dict(text=f"{sym} PREVIEW", font=dict(size=12, color="#7a7a90")),
                        yaxis_tickformat="$,.2f", height=200,
                        margin=dict(l=40, r=20, t=35, b=25),
                    )
                    _phi_chart(fig, height=200)
                st.dataframe(df.tail(10), use_container_width=True)
        return cfg
    return None


# ---------------------------------------------------------------------------
# Step 2 -- Indicator Selection
# ---------------------------------------------------------------------------
def render_indicator_selection():
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
        st.caption("Options mode: Long Call/Put with delta-based simulation.")
        c1, c2, c3 = st.columns(3)
        strat_type = c1.selectbox("Strategy", ["long_call", "long_put"], key="opt_strat")
        exit_profit = c2.slider("Exit profit %", 0.2, 1.0, 0.5, 0.1, key="opt_exit_profit")
        exit_stop = c3.slider("Exit stop %", -0.5, -0.1, -0.3, 0.05, key="opt_exit_stop")
        opts = {"strategy_type": strat_type, "exit_profit_pct": exit_profit, "exit_stop_pct": exit_stop}
        st.session_state["bt_options_controls"] = opts
        return opts


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
def _display_results(config, results, strat, indicators, blend_method, blend_weights, sc=None):
    sc = sc or {}
    tr = _extract_scalar(getattr(results, "total_return", None) or results.get("total_return"))
    cagr = _extract_scalar(getattr(results, "cagr", None) or results.get("cagr"))
    dd = _extract_scalar(getattr(results, "max_drawdown", None) or results.get("max_drawdown"))
    sharpe = _extract_scalar(getattr(results, "sharpe", None) or results.get("sharpe"))
    cap = config.get("initial_capital", 100_000)
    pv = getattr(results, "portfolio_value", None)
    if pv is None: pv = results.get("portfolio_value", [])
    end_cap = pv[-1] if pv and len(pv) else cap
    net_pl = end_cap - cap
    net_pct = (net_pl / cap) * 100 if cap else 0

    st.markdown(_render_section_header("", "RESULTS", "BACKTEST"), unsafe_allow_html=True)

    # Profit/Loss hero display
    pl_color = "#22c55e" if net_pl >= 0 else "#ef4444"
    st.markdown(f"""
    <div style="text-align:center;padding:1.8rem;background:rgba(18,18,26,0.7);border-radius:16px;
                border:1px solid {pl_color}22;margin-bottom:1.5rem;backdrop-filter:blur(16px);">
        <div style="color:#7a7a90;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;font-weight:600;">
            Net Profit / Loss
        </div>
        <div style="color:{pl_color};font-size:2.5rem;font-weight:800;font-family:'JetBrains Mono',monospace;
                    letter-spacing:-0.04em;line-height:1.2;margin-top:0.3rem;">
            ${net_pl:+,.0f}
        </div>
        <div style="color:{pl_color};font-size:1rem;font-weight:600;opacity:0.8;font-family:'JetBrains Mono',monospace;">
            {net_pct:+.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    kpis = [
        ("Start Capital", f"${cap:,.0f}", "", "neutral"),
        ("End Capital", f"${end_cap:,.0f}", "", "positive" if end_cap > cap else "negative"),
        ("CAGR", f"{cagr:+.1%}" if isinstance(cagr, (int, float)) else "--", "", "neutral"),
        ("Sharpe", f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "--", "", "neutral"),
        ("Max Drawdown", f"{dd:.1%}" if isinstance(dd, (int, float)) else "--", "", "negative"),
    ]
    st.markdown(_render_kpi_row(kpis), unsafe_allow_html=True)

    tab_curve, tab_summary, tab_trades, tab_metrics = st.tabs(["Equity Curve", "Summary", "Trades", "Metrics"])

    with tab_curve:
        if pv and len(pv) > 1:
            fig = go.Figure()
            # Color the fill based on performance
            fill_color = "rgba(34,197,94,0.06)" if pv[-1] > cap else "rgba(239,68,68,0.06)"
            line_color = "#22c55e" if pv[-1] > cap else "#ef4444"
            fig.add_trace(go.Scatter(
                y=pv, mode="lines",
                line=dict(color=line_color, width=2.5, shape="spline", smoothing=0.8),
                fill="tozeroy", fillcolor=fill_color,
                name="Portfolio", hovertemplate="$%{y:,.0f}<extra></extra>",
            ))
            fig.add_hline(y=cap, line_dash="dot", line_color="rgba(148,163,184,0.2)",
                          annotation_text=f"Start: ${cap:,.0f}", annotation_font_color="#7a7a90")
            fig.update_layout(
                title=dict(text="EQUITY CURVE", font=dict(size=14, color="#7a7a90")),
                yaxis_title="Portfolio Value ($)", yaxis_tickformat="$,.0f",
            )
            _phi_chart(fig, height=420)

    with tab_summary:
        c1, c2 = st.columns(2)
        c1.metric("Max Drawdown", f"{dd:.1%}" if isinstance(dd, (int, float)) else "--")
        acc = sc.get('accuracy', 0)
        c2.metric("Direction Accuracy", f"{acc:.1%}" if isinstance(acc, (int, float)) else "--")

    with tab_trades:
        if strat and hasattr(strat, "prediction_log") and strat.prediction_log:
            st.dataframe(pd.DataFrame(strat.prediction_log), use_container_width=True)
        else:
            st.caption("No trade log available.")

    with tab_metrics:
        payload = {"total_return": tr, "cagr": cagr, "max_drawdown": dd,
                   "sharpe": sharpe, "accuracy": sc.get("accuracy")}
        if isinstance(results, dict) and results.get("options_snapshot"):
            payload["options_snapshot"] = results.get("options_snapshot")
        st.json(payload)

    from phi.run_config import RunConfig, RunHistory
    run_cfg = RunConfig(
        symbols=config.get("symbols", ["SPY"]),
        start_date=str(config["start"].date()) if hasattr(config["start"], "date") else str(config["start"]),
        end_date=str(config["end"].date()) if hasattr(config["end"], "date") else str(config["end"]),
        timeframe=config.get("timeframe", "1D"), initial_capital=cap,
        indicators=indicators, blend_method=blend_method, blend_weights=blend_weights,
    )
    hist = RunHistory()
    run_id = hist.create_run(run_cfg)
    hist.save_results(run_id, {"total_return": tr, "cagr": cagr, "max_drawdown": dd,
                                "sharpe": sharpe, "accuracy": sc.get("accuracy"), "net_pl": net_pl})
    st.caption(f"Run saved: `{run_id}`")


# ---------------------------------------------------------------------------
# Run & Results Orchestration
# ---------------------------------------------------------------------------
def render_run_and_results(config, indicators, blend_method, blend_weights):
    if not config or not indicators:
        st.info("Complete Steps 1-2 to run a backtest.")
        return

    st.markdown("---")
    st.markdown(_render_section_header("", "RUN BACKTEST", "EXECUTE"), unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    c1.selectbox("Primary metric", METRICS, key="primary_metric")
    run_clicked = c2.button("Run Backtest", type="primary", key="run_bt", use_container_width=True)

    phiai_full = st.session_state.get("phiai_full", False)
    trading_mode = config.get("trading_mode", "equities")

    if not run_clicked:
        return

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
                    phiai_result[0] = run_phiai_optimization(_ohlcv, indicators_to_use, max_iter_per_indicator=15)
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
                        raise ValueError("No data")
                    bt_opts = st.session_state.get("bt_options_controls", {})
                    from phi.options import run_options_backtest
                    opt_result[0] = run_options_backtest(
                        ohlcv, symbol=sym,
                        strategy_type=bt_opts.get("strategy_type", "long_call"),
                        initial_capital=config.get("initial_capital", 100_000),
                        position_pct=0.1,
                        exit_profit_pct=bt_opts.get("exit_profit_pct", 0.5),
                        exit_stop_pct=bt_opts.get("exit_stop_pct", -0.3),
                    )
                except Exception as e:
                    opt_exc[0] = e

            th = threading.Thread(target=run_opt)
            th.start()
            start_t = time.time()
            while th.is_alive():
                time.sleep(0.2)
                pct = min(95, int((time.time() - start_t) * 25))
                opt_progress.progress(pct / 100, text=f"Options backtest... {pct}%")
            if opt_exc[0]: raise opt_exc[0]
            results = opt_result[0]
            opt_progress.progress(1.0, text="Complete!")
            time.sleep(0.3)
            opt_progress.empty()
            if results:
                _display_results(config, results, None, indicators_to_use, blend_method, blend_weights)
            else:
                st.error("No results returned.")
        except Exception as e:
            st.error(str(e))
        return

    # Equities backtest
    use_blended = len(indicators_to_use) >= 2
    progress = st.progress(0, text="Preparing backtest...")

    try:
        if use_blended:
            strat_cls = _load_strategy("strategies.blended_workbench_strategy.BlendedWorkbenchStrategy")
            params = {"symbol": config["symbols"][0], "indicators": indicators_to_use,
                      "blend_method": blend_method, "blend_weights": blend_weights,
                      "signal_threshold": 0.15, "lookback_bars": 200}
        else:
            first_name = list(indicators_to_use.keys())[0]
            info = INDICATOR_CATALOG[first_name]
            strat_cls = _load_strategy(str(info["strategy"]))
            p_defaults = {k: default for k, (_, _, default) in info["params"].items()}
            p_user = {k: int(v) if isinstance(v, float) and v == int(v) else v
                      for k, v in indicators_to_use[first_name].get("params", {}).items()}
            params = {**p_defaults, **p_user, "symbol": config["symbols"][0]}

        result_holder, exc_holder = [None], [None]

        def run_bt():
            try:
                result_holder[0] = _run_backtest(strat_cls, params, config)
            except Exception as e:
                exc_holder[0] = e

        th = threading.Thread(target=run_bt)
        th.start()
        start_t = time.time()
        while th.is_alive():
            time.sleep(0.4)
            pct = min(95, 5 + int((time.time() - start_t) * 1.5))
            progress.progress(pct / 100, text=f"Running backtest... {pct}%")

        if exc_holder[0]: raise exc_holder[0]
        if result_holder[0] and isinstance(result_holder[0], (list, tuple)) and len(result_holder[0]) == 2:
            results, strat = result_holder[0]
            progress.progress(1.0, text="Complete!")
            time.sleep(0.5)
            progress.empty()
            sc = _compute_accuracy(strat) if hasattr(strat, "prediction_log") else {}
            _display_results(config, results, strat, indicators_to_use, blend_method, blend_weights, sc)
        else:
            progress.empty()
            st.error("Backtest returned no results.")

        if phiai_full and phiai_explanation:
            with st.expander("PhiAI Changes"):
                st.markdown(f"""
                <div style="background:rgba(168,85,247,0.04);border:1px solid rgba(168,85,247,0.12);
                            border-radius:12px;padding:1.2rem;font-size:0.88rem;color:#b0b0c0;">
                    {phiai_explanation.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        progress.empty()
        st.error(str(e))
        st.exception(e)


# ---------------------------------------------------------------------------
# AI Agents
# ---------------------------------------------------------------------------
def render_ai_agents():
    st.markdown(_render_section_header("", "AI AGENTS", "OLLAMA"), unsafe_allow_html=True)
    st.caption("Free local models via Ollama. Install: ollama.com/download")

    host = st.text_input("Ollama Host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"), key="ollama_host")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Check connection", use_container_width=True):
            from phi.agents import check_ollama_ready
            if check_ollama_ready(host):
                st.success("Ollama is running")
            else:
                st.error("Ollama not reachable")
    with c2:
        if st.button("List models", use_container_width=True):
            from phi.agents import list_ollama_models
            models = list_ollama_models(host)
            if models:
                names = [m.get("name", "").split(":")[0] for m in models if m.get("name")]
                st.session_state["ollama_models"] = list(dict.fromkeys(names))
                st.success(f"Found {len(st.session_state['ollama_models'])} model(s)")
            else:
                st.warning("No models found")

    model = st.selectbox("Model", options=st.session_state.get("ollama_models", ["llama3.2", "0xroyce/plutus"]),
                          key="ollama_model")
    prompt = st.text_area("Ask AI", placeholder="e.g. What does RSI oversold mean?", key="ollama_prompt", height=80)

    if st.button("Send", key="ollama_send", type="primary") and prompt:
        with st.spinner("Thinking..."):
            try:
                from phi.agents import OllamaAgent
                agent = OllamaAgent(model=model, host=host)
                reply = agent.chat(prompt, system="You are a quantitative trading assistant. Be concise.")
                st.markdown("**Reply:**")
                st.markdown(reply)
            except Exception as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Run History & Cache
# ---------------------------------------------------------------------------
def render_run_history():
    from phi.run_config import RunHistory
    hist = RunHistory()
    runs = hist.list_runs()
    if not runs:
        st.caption("No runs yet.")
        return
    st.markdown(_render_section_header("", "RUN HISTORY", f"{len(runs)} RUNS"), unsafe_allow_html=True)
    for r in runs[:10]:
        with st.expander(r["run_id"]):
            st.json(r.get("results", {}))


def render_cache_manager():
    from phi.data import list_cached_datasets
    datasets = list_cached_datasets()
    st.markdown(_render_section_header("", "CACHE MANAGER", ""), unsafe_allow_html=True)
    if not datasets:
        st.caption("No cached datasets.")
        return
    st.dataframe(pd.DataFrame(datasets), use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Phi-nance | Live Workbench",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
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

    st.markdown("---")
    tab_hist, tab_cache, tab_agents = st.tabs(["Run History", "Cache Manager", "AI Agents"])
    with tab_hist: render_run_history()
    with tab_cache: render_cache_manager()
    with tab_agents: render_ai_agents()

    # Footer
    extra_pad = "padding-bottom:4rem;" if dev.is_mobile else ""
    st.markdown(f"""
    <div class="phi-footer" style="{extra_pad}">
        PHI-NANCE &middot; LIVE BACKTEST WORKBENCH &middot; v3.1 PREMIUM
        <br>QUANTITATIVE RESEARCH PLATFORM &middot; {datetime.now().strftime("%Y")}
        <br><span style="font-size:0.55rem;opacity:0.5;">
            {dev.device_type.value.upper()} MODE &middot; {dev.screen_width}px
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phi-nance — One page. Everything works.
"""

import os, sys, math, warnings
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("IS_BACKTESTING", "True")
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Phi-nance", page_icon="φ", layout="wide")

# ── Minimal clean CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0a0a0f; }
    section[data-testid="stSidebar"] { display: none; }

    /* Cards */
    .card {
        background: #14141e; border: 1px solid #1e1e30; border-radius: 10px;
        padding: 18px; margin-bottom: 12px;
    }
    .card h3 { margin: 0 0 4px 0; font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .card .val { font-size: 1.6rem; font-weight: 700; color: #e8e8ed; }
    .card .sub { font-size: 0.8rem; color: #666; margin-top: 2px; }
    .green { color: #22c55e !important; }
    .red { color: #ef4444 !important; }
    .purple { color: #a855f7 !important; }

    /* Signal badge */
    .signal { display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; }
    .signal-buy { background: #22c55e22; color: #22c55e; border: 1px solid #22c55e44; }
    .signal-sell { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }
    .signal-hold { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }

    /* Tweak card */
    .tweak { background: #14141e; border: 1px solid #a855f733; border-radius: 8px; padding: 12px; margin: 6px 0; }
    .tweak .title { color: #a855f7; font-weight: 700; }
    .tweak .body { color: #999; font-size: 0.85rem; margin-top: 4px; }

    /* Section divider */
    .divider { border-top: 1px solid #1e1e30; margin: 24px 0; }

    /* Hide default header/footer */
    header[data-testid="stHeader"] { background: transparent; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_data(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Load OHLCV from yfinance. Returns lowercase columns, datetime index."""
    try:
        import yfinance as yf
        raw = yf.download(symbol, start=start, end=end, progress=False)
        if raw is None or raw.empty:
            return None
        # Handle multi-level columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        # Ensure all required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in raw.columns:
                return None
        raw = raw[["open", "high", "low", "close", "volume"]].dropna()
        raw.index = pd.to_datetime(raw.index)
        return raw
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════
# CHART BUILDERS (simple, reliable plotly)
# ═════════════════════════════════════════════════════════════════════════

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,14,20,0.6)",
    font=dict(color="#ccc", size=11),
    margin=dict(l=50, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#1a1a2a"),
    yaxis=dict(gridcolor="#1a1a2a"),
)


def chart_equity(pv, capital):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=pv, mode="lines", name="Portfolio",
        line=dict(color="#a855f7", width=2),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.08)",
    ))
    fig.add_hline(y=capital, line_dash="dot", line_color="#444", annotation_text="Start")
    fig.update_layout(**_LAYOUT, height=350, yaxis_title="Value ($)")
    return fig


def chart_candles(df, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        name=symbol,
    )])
    fig.update_layout(**_LAYOUT, height=300, xaxis_rangeslider_visible=False)
    return fig


def chart_signals(df, composite, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["close"], mode="lines", name=symbol,
        line=dict(color="#888", width=1),
    ))
    # Color signal: green when > 0, red when < 0
    fig.add_trace(go.Scatter(
        x=df.index, y=composite.reindex(df.index), mode="lines", name="Signal",
        line=dict(color="#a855f7", width=2), yaxis="y2",
    ))
    fig.update_layout(
        **_LAYOUT, height=300,
        yaxis2=dict(overlaying="y", side="right", gridcolor="#1a1a2a",
                    title="Signal", range=[-1.2, 1.2]),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════

st.markdown("# φ Phi-nance")
st.caption("Get a trade idea. Run a backtest. Wire PhiBot into your trading.")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETUP (always visible)
# ═════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
with c1:
    symbol = st.text_input("Ticker", value="SPY").strip().upper()
with c2:
    start_date = st.date_input("From", value=date(2023, 1, 1))
with c3:
    end_date = st.date_input("To", value=date(2025, 1, 1))
with c4:
    capital = st.number_input("Capital ($)", value=100_000, min_value=1_000, step=10_000)

# Load data immediately
ohlcv = load_data(symbol, str(start_date), str(end_date))
if ohlcv is None or ohlcv.empty:
    st.error(f"No data for **{symbol}** in that range. Try SPY, AAPL, QQQ, etc.")
    st.stop()

spot = float(ohlcv["close"].iloc[-1])
change_pct = (ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1) * 100

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown(f'<div class="card"><h3>Last Price</h3><div class="val">${spot:,.2f}</div>'
                f'<div class="sub">{symbol}</div></div>', unsafe_allow_html=True)
with mc2:
    color = "green" if change_pct >= 0 else "red"
    st.markdown(f'<div class="card"><h3>Period Return</h3><div class="val {color}">{change_pct:+.1f}%</div>'
                f'<div class="sub">{len(ohlcv)} bars</div></div>', unsafe_allow_html=True)
with mc3:
    # Quick vol calc
    rets = np.diff(np.log(ohlcv["close"].values.astype(float)))
    ann_vol = float(np.std(rets) * np.sqrt(252) * 100) if len(rets) > 5 else 0
    st.markdown(f'<div class="card"><h3>Annualized Vol</h3><div class="val">{ann_vol:.1f}%</div>'
                f'<div class="sub">21d realized</div></div>', unsafe_allow_html=True)

st.plotly_chart(chart_candles(ohlcv, symbol), use_container_width=True, key="candles")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# SECTION 2 — GET A TRADE IDEA (AI Advisor)
# ═════════════════════════════════════════════════════════════════════════

st.markdown("## 🧠 Get a Trade Idea")
st.caption("PhiBot analyzes the current market regime, volatility, and momentum to suggest a trade.")

idea_c1, idea_c2 = st.columns([1, 2])

with idea_c1:
    idea_mode = st.radio("Mode", ["Stocks", "Options"], horizontal=True, key="idea_mode")

if st.button("🧠 What should I trade?", type="primary", use_container_width=True, key="get_idea"):
    with st.spinner("Analyzing..."):
        close_arr = ohlcv["close"].values.astype(float)

        # Compute indicators
        from phi.indicators.simple import compute_rsi, compute_macd, compute_bollinger, compute_dual_sma
        rsi_sig = compute_rsi(ohlcv)
        macd_sig = compute_macd(ohlcv)
        boll_sig = compute_bollinger(ohlcv)
        sma_sig = compute_dual_sma(ohlcv)

        # Blend
        signals_df = pd.DataFrame({
            "RSI": rsi_sig, "MACD": macd_sig,
            "Bollinger": boll_sig, "Dual SMA": sma_sig,
        })
        from phi.blending.blender import blend_signals
        composite = blend_signals(signals_df, method="weighted_sum")
        latest = float(composite.iloc[-1]) if not composite.empty else 0.0

        # Regime detection
        try:
            from regime_engine import RegimeEngine
            re = RegimeEngine()
            out = re.run(ohlcv)
            rp = out.get("regime_probs")
            if rp is not None and not rp.empty:
                regime_probs = rp.iloc[-1].to_dict()
            else:
                regime_probs = {"RANGE": 0.5, "TREND_UP": 0.25, "TREND_DN": 0.25}
        except Exception:
            regime_probs = {"RANGE": 0.5, "TREND_UP": 0.25, "TREND_DN": 0.25}

        dominant_regime = max(regime_probs, key=regime_probs.get)
        regime_conf = regime_probs.get(dominant_regime, 0.5)

        # Historical vol
        if len(close_arr) > 22:
            hv = float(np.std(np.diff(np.log(np.maximum(close_arr[-22:], 1e-10)))) * np.sqrt(252))
        else:
            hv = 0.20

        if idea_mode == "Stocks":
            # Simple stock signal
            if latest > 0.15:
                signal, direction, reasoning = "BUY", "BULLISH", (
                    f"Composite signal is **{latest:+.2f}** (bullish). "
                    f"Dominant regime: **{dominant_regime}** ({regime_conf:.0%} confidence). "
                    f"RSI, MACD, Bollinger, and SMA are leaning positive."
                )
            elif latest < -0.15:
                signal, direction, reasoning = "SELL", "BEARISH", (
                    f"Composite signal is **{latest:+.2f}** (bearish). "
                    f"Dominant regime: **{dominant_regime}** ({regime_conf:.0%} confidence). "
                    f"Indicators are pointing down — consider reducing exposure."
                )
            else:
                signal, direction, reasoning = "HOLD", "NEUTRAL", (
                    f"Composite signal is **{latest:+.2f}** (neutral). "
                    f"Dominant regime: **{dominant_regime}** ({regime_conf:.0%} confidence). "
                    f"No strong edge — wait for a clearer setup."
                )

            risk_note = (
                f"Annualized vol is {hv*100:.0f}%. "
                f"{'High volatility — size down.' if hv > 0.25 else 'Vol is normal.'} "
                f"Always use a stop loss."
            )

            st.session_state["idea"] = {
                "mode": "Stocks", "signal": signal, "direction": direction,
                "reasoning": reasoning, "risk_note": risk_note,
                "composite": latest, "regime": dominant_regime,
                "regime_conf": regime_conf, "hv": hv,
                "composite_series": composite,
            }

        else:
            # Options — use the AI advisor
            from phi.options.ai_advisor import OptionsAIAdvisor
            advisor = OptionsAIAdvisor(use_ollama=False)

            # Classify IV regime
            iv_est = hv * 1.15
            if hv > 0:
                ratio = iv_est / hv
                iv_regime = "HIGH_IV" if ratio >= 1.3 else ("LOW_IV" if ratio <= 0.8 else "NORMAL")
            else:
                iv_regime = "NORMAL"

            gamma_features = {"gamma_net": 0.0, "gex_flip_zone": 0.0,
                              "gamma_wall_distance": 0.0, "gamma_expiry_days": 30.0}

            rec = advisor.recommend(
                symbol=symbol, spot=spot, hist_vol=hv,
                regime_probs=regime_probs,
                gamma_features=gamma_features,
                iv_regime=iv_regime,
            )

            signal_map = {"ENTER": "BUY", "WAIT": "HOLD", "SKIP": "SELL"}
            signal = signal_map.get(rec.entry_signal, "HOLD")

            st.session_state["idea"] = {
                "mode": "Options", "signal": signal,
                "direction": rec.direction,
                "structure": rec.structure.replace("_", " ").title(),
                "level": rec.level,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "risk_note": rec.risk_note,
                "position_size": rec.position_size,
                "regime": rec.regime,
                "vol_regime": rec.vol_regime,
                "gex_regime": rec.gex_regime,
                "composite_series": composite,
                "hv": hv,
            }

# Display idea
idea = st.session_state.get("idea")
if idea:
    st.markdown("")

    # Signal badge
    badge_cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[idea["signal"]]
    badge_html = f'<span class="signal {badge_cls}">{idea["signal"]}</span>'

    if idea["mode"] == "Stocks":
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.markdown(f'<div class="card"><h3>Signal</h3><div style="margin-top:8px">{badge_html}</div></div>',
                        unsafe_allow_html=True)
        with ic2:
            st.markdown(f'<div class="card"><h3>Direction</h3><div class="val">{idea["direction"]}</div></div>',
                        unsafe_allow_html=True)
        with ic3:
            st.markdown(f'<div class="card"><h3>Regime</h3><div class="val">{idea["regime"]}</div>'
                        f'<div class="sub">{idea["regime_conf"]:.0%} confidence</div></div>',
                        unsafe_allow_html=True)
        with ic4:
            st.markdown(f'<div class="card"><h3>Signal Strength</h3><div class="val">{idea["composite"]:+.2f}</div>'
                        f'<div class="sub">-1 to +1 scale</div></div>',
                        unsafe_allow_html=True)
    else:
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.markdown(f'<div class="card"><h3>Signal</h3><div style="margin-top:8px">{badge_html}</div></div>',
                        unsafe_allow_html=True)
        with ic2:
            st.markdown(f'<div class="card"><h3>Structure</h3><div class="val">{idea["structure"]}</div>'
                        f'<div class="sub">{idea["level"]} — {idea["direction"]}</div></div>',
                        unsafe_allow_html=True)
        with ic3:
            st.markdown(f'<div class="card"><h3>Confidence</h3><div class="val purple">{idea["confidence"]:.0%}</div>'
                        f'<div class="sub">Position: {idea["position_size"]:.0%} of capital</div></div>',
                        unsafe_allow_html=True)
        with ic4:
            st.markdown(f'<div class="card"><h3>Regime</h3><div class="val">{idea["regime"]}</div>'
                        f'<div class="sub">IV: {idea["vol_regime"]} · GEX: {idea["gex_regime"]}</div></div>',
                        unsafe_allow_html=True)

    st.markdown(f"**Why:** {idea['reasoning']}")
    st.markdown(f"**Risk:** {idea['risk_note']}")

    # Show signal chart
    comp = idea.get("composite_series")
    if comp is not None and not comp.empty:
        st.plotly_chart(chart_signals(ohlcv, comp, symbol), use_container_width=True, key="sig_chart")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# SECTION 3 — BACKTEST (stocks or options)
# ═════════════════════════════════════════════════════════════════════════

st.markdown("## 📊 Backtest")
st.caption("Pick your indicators and blend method, then hit Run to see how it would have performed.")

bt_tab1, bt_tab2 = st.tabs(["Stock Backtest", "Options Backtest"])

# ── Stock Backtest ────────────────────────────────────────────────────────
with bt_tab1:
    s_c1, s_c2 = st.columns([1, 1])
    with s_c1:
        avail_indicators = ["RSI", "MACD", "Bollinger", "Dual SMA", "Mean Reversion", "Breakout"]
        chosen_indicators = st.multiselect(
            "Indicators", avail_indicators, default=["RSI", "MACD", "Bollinger"],
            key="bt_indicators",
        )
        blend_method = st.selectbox(
            "Blend method", ["weighted_sum", "voting", "regime_weighted"],
            key="bt_blend",
        )
    with s_c2:
        signal_thresh = st.slider("Signal threshold", 0.05, 0.50, 0.15, 0.05,
                                  help="Signal must exceed this to trigger a trade", key="bt_thresh")
        position_pct = st.slider("Position size (%)", 10, 100, 95, 5, key="bt_pospct") / 100.0

    if st.button("🚀 Run Stock Backtest", type="primary", use_container_width=True, key="run_stock_bt"):
        if not chosen_indicators:
            st.error("Pick at least one indicator.")
        else:
            with st.spinner("Backtesting..."):
                from phi.backtest.direct import run_direct_backtest
                indicators = {name: {"params": {}} for name in chosen_indicators}
                weights = {name: 1.0 / len(chosen_indicators) for name in chosen_indicators}
                results, strat = run_direct_backtest(
                    ohlcv=ohlcv, symbol=symbol,
                    indicators=indicators,
                    blend_weights=weights,
                    blend_method=blend_method,
                    signal_threshold=signal_thresh,
                    initial_capital=float(capital),
                    position_size_pct=position_pct,
                )
                st.session_state["stock_bt"] = results

    sbt = st.session_state.get("stock_bt")
    if sbt:
        pv = sbt.get("portfolio_value", [])
        tr = sbt.get("total_return", 0)
        cagr = sbt.get("cagr", 0)
        sharpe = sbt.get("sharpe", 0)
        dd = sbt.get("max_drawdown", 0)
        net = sbt.get("net_pl", 0)

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            c = "green" if tr >= 0 else "red"
            st.markdown(f'<div class="card"><h3>Total Return</h3><div class="val {c}">{tr*100:+.1f}%</div></div>',
                        unsafe_allow_html=True)
        with r2:
            c = "green" if cagr >= 0 else "red"
            st.markdown(f'<div class="card"><h3>CAGR</h3><div class="val {c}">{cagr*100:+.1f}%</div></div>',
                        unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="card"><h3>Sharpe</h3><div class="val">{sharpe:.2f}</div></div>',
                        unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="card"><h3>Max Drawdown</h3><div class="val red">{dd*100:.1f}%</div></div>',
                        unsafe_allow_html=True)
        with r5:
            c = "green" if net >= 0 else "red"
            st.markdown(f'<div class="card"><h3>Net P&L</h3><div class="val {c}">${net:+,.0f}</div></div>',
                        unsafe_allow_html=True)

        if len(pv) > 2:
            st.plotly_chart(chart_equity(pv, float(capital)), use_container_width=True, key="stock_eq")


# ── Options Backtest ──────────────────────────────────────────────────────
with bt_tab2:
    o_c1, o_c2, o_c3 = st.columns(3)
    with o_c1:
        opt_conf = st.slider("Min confidence", 0.20, 0.80, 0.35, 0.05, key="opt_conf")
        opt_pos = st.slider("Position size (%)", 1, 30, 10, 1, key="opt_pos") / 100.0
    with o_c2:
        opt_dte = st.slider("Target DTE", 14, 90, 30, 1, key="opt_dte")
        opt_iv_prem = st.slider("IV premium", 1.00, 2.00, 1.15, 0.05, key="opt_iv")
    with o_c3:
        opt_profit = st.slider("Profit target (%)", 10, 200, 50, 10, key="opt_profit") / 100.0
        opt_stop = st.slider("Stop loss (%)", 10, 100, 30, 10, key="opt_stop") / 100.0

    if st.button("🚀 Run Options Backtest", type="primary", use_container_width=True, key="run_opt_bt"):
        with st.spinner("Running options engine backtest..."):
            try:
                from phi.options.engine_backtest import run_engine_backtest
                prog = st.progress(0.0, text="Starting...")

                def _prog(f):
                    prog.progress(min(f, 1.0), text=f"Backtest: {f*100:.0f}%")

                results = run_engine_backtest(
                    ohlcv=ohlcv, symbol=symbol,
                    initial_capital=float(capital),
                    position_pct=opt_pos,
                    min_confidence=opt_conf,
                    dte_days=opt_dte,
                    exit_profit_pct=opt_profit,
                    exit_stop_pct=opt_stop,
                    iv_premium=opt_iv_prem,
                    progress_cb=_prog,
                )
                prog.progress(1.0, text="Done!")
                st.session_state["opt_bt"] = results
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    obt = st.session_state.get("opt_bt")
    if obt:
        pv = obt.get("portfolio_value", [])
        tr = obt.get("total_return", 0)
        cagr = obt.get("cagr", 0)
        sharpe = obt.get("sharpe", 0)
        dd = obt.get("max_drawdown", 0)
        wr = obt.get("win_rate", 0)
        nt = obt.get("n_trades", 0)

        # Status flags
        flags = {
            "RegimeEngine": obt.get("_regime_engine_ok", False),
            "OptionsEngine": obt.get("_options_ok", False),
            "GammaSurface": obt.get("_gamma_ok", False),
            "AI Advisor": obt.get("_ai_advisor_ok", False),
        }
        flag_html = "  ".join(
            f'<span style="color:{"#22c55e" if ok else "#555"}">{"●" if ok else "○"} {n}</span>'
            for n, ok in flags.items()
        )
        st.markdown(f'<div style="font-size:0.8rem; margin-bottom:8px;">{flag_html}</div>', unsafe_allow_html=True)

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1:
            c = "green" if tr >= 0 else "red"
            st.markdown(f'<div class="card"><h3>Total Return</h3><div class="val {c}">{tr*100:+.1f}%</div></div>',
                        unsafe_allow_html=True)
        with r2:
            c = "green" if cagr >= 0 else "red"
            st.markdown(f'<div class="card"><h3>CAGR</h3><div class="val {c}">{cagr*100:+.1f}%</div></div>',
                        unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="card"><h3>Sharpe</h3><div class="val">{sharpe:.2f}</div></div>',
                        unsafe_allow_html=True)
        with r4:
            st.markdown(f'<div class="card"><h3>Max Drawdown</h3><div class="val red">{dd*100:.1f}%</div></div>',
                        unsafe_allow_html=True)
        with r5:
            st.markdown(f'<div class="card"><h3>Win Rate</h3><div class="val">{wr*100:.0f}%</div>'
                        f'<div class="sub">{nt} trades</div></div>', unsafe_allow_html=True)

        if len(pv) > 2:
            st.plotly_chart(chart_equity(pv, float(capital)), use_container_width=True, key="opt_eq")

        # Trade log
        trades = obt.get("trades", [])
        if trades:
            with st.expander(f"Trade Log ({len(trades)} trades)", expanded=False):
                df_t = pd.DataFrame(trades)
                show = [c for c in [
                    "entry_date", "exit_date", "structure", "regime",
                    "confidence", "days_held", "exit_reason", "pnl_$",
                ] if c in df_t.columns]
                st.dataframe(df_t[show], hide_index=True, use_container_width=True)

        # AI Review
        if trades:
            with st.expander("🤖 PhiBot Review", expanded=True):
                try:
                    from phi.options.options_reviewer import review_options_backtest
                    review = review_options_backtest(trade_log=trades, metrics=obt, ohlcv=ohlcv)

                    verdict_colors = {"strong": "#22c55e", "moderate": "#f59e0b", "weak": "#ef4444", "neutral": "#888"}
                    vc = verdict_colors.get(review.verdict, "#888")
                    st.markdown(
                        f'<span style="background:{vc}22; color:{vc}; padding:3px 12px; border-radius:12px; '
                        f'font-weight:700; border:1px solid {vc}44;">{review.verdict.upper()}</span> '
                        f'<span style="color:#ccc;">{review.summary}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                    for ob in review.observations:
                        st.markdown(f"- {ob}")

                    if review.tweaks:
                        st.markdown("**Suggested tweaks:**")
                        for twk in review.tweaks:
                            st.markdown(
                                f'<div class="tweak"><div class="title">{twk.title}</div>'
                                f'<div class="body">{twk.rationale}<br>'
                                f'<em>{twk.current_value} → {twk.suggested_value}</em></div></div>',
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.warning(f"Review unavailable: {e}")


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# SECTION 4 — WIRE PHIBOT INTO YOUR TRADING
# ═════════════════════════════════════════════════════════════════════════

st.markdown("## 🔌 Wire PhiBot Into Your Trading")
st.caption("Connect PhiBot to a local Ollama LLM for smarter, AI-powered recommendations.")

wire_c1, wire_c2 = st.columns([1, 1])

with wire_c1:
    st.markdown("""
    **Quick setup (3 steps):**
    1. Install [Ollama](https://ollama.com/download)
    2. Pull a model: `ollama pull llama3.2`
    3. Start it: `ollama serve`

    PhiBot will automatically use Ollama for richer analysis when available.
    For finance-specific reasoning, try: `ollama pull 0xroyce/plutus`
    """)

with wire_c2:
    ollama_host = st.text_input("Ollama host", value="http://localhost:11434", key="ollama_host")
    ollama_model = st.text_input("Model", value="llama3.2", key="ollama_model")

    if st.button("🔍 Test Connection", use_container_width=True, key="test_ollama"):
        try:
            from phi.agents import check_ollama_ready
            if check_ollama_ready(ollama_host):
                st.success("Connected! PhiBot will use Ollama for AI-powered analysis.")
                from phi.agents import list_ollama_models
                models = list_ollama_models(ollama_host)
                if models:
                    st.info(f"Available models: {', '.join(models)}")
            else:
                st.warning("Ollama not reachable. PhiBot will use rule-based analysis (still works great).")
        except Exception as e:
            st.warning(f"Can't reach Ollama ({e}). Rule-based mode active — still works.")

    if st.button("🧠 Ask PhiBot (Ollama)", use_container_width=True, key="ask_phibot"):
        with st.spinner("Asking PhiBot..."):
            try:
                from phi.options.ai_advisor import OptionsAIAdvisor
                advisor = OptionsAIAdvisor(
                    model=ollama_model, host=ollama_host, use_ollama=True,
                )

                close_arr = ohlcv["close"].values.astype(float)
                if len(close_arr) > 22:
                    hv = float(np.std(np.diff(np.log(np.maximum(close_arr[-22:], 1e-10)))) * np.sqrt(252))
                else:
                    hv = 0.20

                regime_probs = {"RANGE": 0.4, "TREND_UP": 0.3, "TREND_DN": 0.2, "HIGHVOL": 0.1}
                gamma_features = {"gamma_net": 0.0, "gex_flip_zone": 0.0,
                                  "gamma_wall_distance": 0.0, "gamma_expiry_days": 30.0}

                rec = advisor.recommend(
                    symbol=symbol, spot=spot, hist_vol=hv,
                    regime_probs=regime_probs,
                    gamma_features=gamma_features,
                )

                source_label = "🤖 AI (Ollama)" if rec.source == "ai" else "📐 Rules (Ollama offline)"
                st.success(f"**{source_label}** → {rec.structure.replace('_', ' ').title()} "
                           f"({rec.direction}, {rec.confidence:.0%} confidence, {rec.entry_signal})")
                st.markdown(f"**Why:** {rec.reasoning}")
                st.markdown(f"**Risk:** {rec.risk_note}")

            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 9 — Advanced Pages
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")

_ADV_PAGES = {
    "🤖 Autonomous Pipeline":      "autonomous",
    "📡 Live / Paper Trading":     "live",
    "🔌 Plugin Browser":           "plugins",
    "🧬 Evolution Dashboard":      "evolution",
    "💼 Portfolio Backtest":       "portfolio",
}

_adv_page = st.radio(
    "**Advanced Tools**",
    list(_ADV_PAGES.keys()),
    horizontal=True,
    index=None,
    key="adv_page_radio",
)

if _adv_page == "🤖 Autonomous Pipeline":
    try:
        from app_streamlit.pages.autonomous_pipeline_page import render_autonomous_pipeline
        render_autonomous_pipeline()
    except Exception as _e:
        st.error(f"Autonomous Pipeline error: {_e}")

elif _adv_page == "📡 Live / Paper Trading":
    try:
        from app_streamlit.pages.live_trading_dashboard import render_live_trading_dashboard
        render_live_trading_dashboard()
    except Exception as _e:
        st.error(f"Live Trading error: {_e}")

elif _adv_page == "🔌 Plugin Browser":
    try:
        from app_streamlit.pages.plugin_browser import render_plugin_browser
        render_plugin_browser()
    except Exception as _e:
        st.error(f"Plugin Browser error: {_e}")

elif _adv_page == "🧬 Evolution Dashboard":
    try:
        from app_streamlit.pages.evolution_dashboard import render as render_evolution
        render_evolution()
    except Exception as _e:
        st.error(f"Evolution Dashboard error: {_e}")

elif _adv_page == "💼 Portfolio Backtest":
    try:
        from app_streamlit.pages.portfolio_backtest_page import render as render_portfolio
        render_portfolio()
    except Exception as _e:
        st.error(f"Portfolio Backtest error: {_e}")


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("φ Phi-nance · Market Field Theory · Built for regular people who want an edge.")

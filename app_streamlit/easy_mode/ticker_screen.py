"""Single-ticker price + signal fields + regime ribbon."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from phi.indicators.simple import compute_indicator
from phi.options.regime_playbook import playbook_entry_for_label, quick_detailed_regime_from_ohlcv
from phi.regime.train import train_regime_detector

from app_streamlit.easy_mode.constants import LOOKBACK_DAYS, UNIVERSE
from app_streamlit.easy_mode.data import load_ohlcv
from app_streamlit.easy_mode.time_helpers import seconds_until_next_us_equity_daily_close


def render_ticker_spotlight() -> None:
    st.markdown('<p class="phinance-hero">Ticker spotlight</p>', unsafe_allow_html=True)
    st.caption("Price, detected regime lane, and two normalized signal tracks.")

    sec, close_lbl = seconds_until_next_us_equity_daily_close()
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    st.metric(
        "Countdown to next US cash close (daily bar context)",
        f"{h}h {m}m {s}s",
        help=f"Next weekday 4pm ET target: {close_lbl}. Weekends roll to Monday.",
    )

    sym = st.selectbox("Ticker", options=list(UNIVERSE), index=0, key="easy_ticker_pick")

    try:
        ohlcv = load_ohlcv(sym, LOOKBACK_DAYS)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return

    composite = quick_detailed_regime_from_ohlcv(ohlcv)
    pb = playbook_entry_for_label(composite)
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot regime (quick)", composite)
    c2.metric("Last close", f"{float(ohlcv['close'].iloc[-1]):.2f}")
    if pb and pb.allowed_structures:
        focus = pb.allowed_structures[0]
        if len(focus) > 28:
            focus = focus[:25] + "…"
        c3.metric("Playbook focus", focus)
    else:
        c3.metric("Playbook focus", "—")

    with st.spinner("Labeling bars with regime model…"):
        det, _ = train_regime_detector(ohlcv, method="kmeans", n_regimes=3, window=20, save=False)
        regimes = det.predict(ohlcv)

    rsi_sig = compute_indicator("RSI", ohlcv, {"rsi_period": 14, "oversold": 30, "overbought": 70})
    macd_sig = compute_indicator("MACD", ohlcv, {"fast_period": 12, "slow_period": 26, "signal_period": 9})

    reg_cat = pd.Categorical(regimes.astype(str))
    reg_codes = pd.Series(reg_cat.codes, index=regimes.index).astype(float)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.42, 0.16, 0.21, 0.21],
        subplot_titles=("Price", "Regime state (0,1,2)", "RSI signal (−1…1)", "MACD signal (−1…1)"),
    )
    fig.add_trace(
        go.Scatter(x=ohlcv.index, y=ohlcv["close"], name="Close", line=dict(color="#38bdf8", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv.index,
            y=reg_codes,
            name="Regime",
            line=dict(color="#a78bfa", width=1.5),
            mode="lines",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=ohlcv.index, y=rsi_sig, name="RSI signal", line=dict(color="#f472b6", width=1.2)),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=3, col=1)

    fig.add_trace(
        go.Scatter(x=ohlcv.index, y=macd_sig, name="MACD signal", line=dict(color="#34d399", width=1.2)),
        row=4,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=820,
        paper_bgcolor="rgba(15,23,42,0.7)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        showlegend=True,
        margin=dict(l=40, r=30, t=48, b=40),
        title=f"{sym} — what’s going on",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Regime lane is an on-the-fly 3-state model for this chart window (not your saved expert models).")

"""
Page 5 — Backtest Controls
===========================

Configure and launch the backtest.
Stores BacktestResult in st.session_state["backtest_result"].
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phinance.backtest import run_backtest
from phinance.storage import RunHistory
from phinance.config.run_config import RunConfig

st.set_page_config(page_title="Backtest Controls | Phi-nance", layout="wide")
st.title("5 · Backtest Controls")
st.caption("Configure execution settings and launch the backtest.")

ohlcv       = st.session_state.get("ohlcv")
indicators  = st.session_state.get("indicators", {})
blend_method= st.session_state.get("blend_method", "weighted_sum")
blend_weights=st.session_state.get("blend_weights", {})
symbol      = st.session_state.get("symbol", "SPY")
timeframe   = st.session_state.get("timeframe", "1D")

if ohlcv is None or ohlcv.empty:
    st.warning("No dataset loaded. Go back to Step 1.")
    st.stop()

active = {n: v for n, v in indicators.items() if v.get("enabled")}
if not active:
    st.warning("No active indicators. Go back to Step 2.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1_000.0, max_value=10_000_000.0,
        value=float(st.session_state.get("initial_capital", 100_000)),
        step=10_000.0,
        format="%.0f",
    )
    trading_mode = st.radio(
        "Trading Mode", ["equities", "options"],
        index=["equities", "options"].index(
            st.session_state.get("trading_mode", "equities")
        ),
        horizontal=True,
    )
with col2:
    signal_threshold = st.slider(
        "Signal Threshold",
        min_value=0.05, max_value=0.5,
        value=float(st.session_state.get("signal_threshold", 0.15)),
        step=0.01,
        help="Minimum signal magnitude required to enter/exit a position.",
    )
    position_size = st.slider(
        "Position Size (%)",
        min_value=10, max_value=100, value=95, step=5,
        help="Fraction of capital deployed per trade.",
    )

# ── Summary ───────────────────────────────────────────────────────────────────
st.subheader("Run Summary")
import pandas as pd

st.dataframe(
    pd.DataFrame([
        {"Setting": "Symbol",         "Value": symbol},
        {"Setting": "Timeframe",      "Value": timeframe},
        {"Setting": "Active Inds.",   "Value": ", ".join(active.keys())},
        {"Setting": "Blend Method",   "Value": blend_method},
        {"Setting": "Initial Capital","Value": f"${initial_capital:,.0f}"},
        {"Setting": "Signal Threshold","Value": f"{signal_threshold:.2f}"},
        {"Setting": "Position Size",  "Value": f"{position_size}%"},
        {"Setting": "Trading Mode",   "Value": trading_mode},
    ]),
    use_container_width=True, hide_index=True,
)

# ── Launch ────────────────────────────────────────────────────────────────────
if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
    st.session_state.update({
        "initial_capital":  initial_capital,
        "signal_threshold": signal_threshold,
        "trading_mode":     trading_mode,
    })

    with st.spinner("Running backtest…"):
        try:
            if trading_mode == "options":
                from phinance.options.backtest import run_options_backtest
                raw = run_options_backtest(
                    ohlcv           = ohlcv,
                    symbol          = symbol,
                    strategy_type   = "long_call",
                    initial_capital = initial_capital,
                )
                # Convert raw dict to BacktestResult-like for unified display
                from phinance.backtest.models import BacktestResult
                result = BacktestResult(
                    symbol          = symbol,
                    total_return    = raw.get("total_return", 0),
                    cagr            = raw.get("cagr", 0),
                    max_drawdown    = abs(raw.get("max_drawdown", 0)),
                    sharpe          = raw.get("sharpe", 0),
                    portfolio_value = raw.get("portfolio_value", [initial_capital]),
                    metadata        = raw,
                )
            else:
                result = run_backtest(
                    ohlcv              = ohlcv,
                    symbol             = symbol,
                    indicators         = active,
                    blend_weights      = blend_weights,
                    blend_method       = blend_method,
                    signal_threshold   = signal_threshold,
                    initial_capital    = initial_capital,
                    position_size_pct  = position_size / 100,
                )

            # Persist run
            cfg = RunConfig(
                symbols          = [symbol],
                start_date       = st.session_state.get("start_date", ""),
                end_date         = st.session_state.get("end_date", ""),
                timeframe        = timeframe,
                vendor           = st.session_state.get("vendor", "alphavantage"),
                initial_capital  = initial_capital,
                trading_mode     = trading_mode,
                indicators       = active,
                blend_method     = blend_method,
                blend_weights    = blend_weights,
            )
            history = RunHistory()
            run_id  = history.create_run(cfg)
            history.save_results(run_id, result.to_dict())

            st.session_state["backtest_result"] = result
            st.session_state["run_id"] = run_id
            st.success(f"Backtest complete! Run ID: `{run_id}`")
            st.balloons()

        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            import traceback
            st.code(traceback.format_exc())

st.divider()
st.markdown("**Next →** [6 · Results](6_Results)")

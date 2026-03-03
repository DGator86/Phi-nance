"""
Page 4 — PhiAI Optimization
============================

Run PhiAI auto-tuning over active indicator parameter grids.
Updates st.session_state["indicators"] with optimised params.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phinance.optimization import run_phiai_optimization, PhiAI

st.set_page_config(page_title="PhiAI Optimization | Phi-nance", layout="wide")
st.title("4 · PhiAI Optimization")
st.caption("Auto-tune indicator parameters using random search on directional accuracy.")

indicators = st.session_state.get("indicators", {})
active     = {n: v for n, v in indicators.items() if v.get("enabled")}
ohlcv      = st.session_state.get("ohlcv")
timeframe  = st.session_state.get("timeframe", "1D")

if not active:
    st.warning("No active indicators. Go back to Step 2 and enable at least one.")
    st.stop()

if ohlcv is None or ohlcv.empty:
    st.warning("No dataset loaded. Go back to Step 1 and fetch data.")
    st.stop()

# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("PhiAI Settings")
    max_iter = st.slider(
        "Max iterations per indicator",
        min_value=5, max_value=100, value=20, step=5,
    )
    max_indicators = st.number_input("Max indicators", min_value=1, max_value=10, value=5)
    allow_shorts   = st.checkbox("Allow shorts", value=False)
    risk_cap_input = st.number_input("Risk cap (0 = none)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    risk_cap = risk_cap_input if risk_cap_input > 0 else None

ai = PhiAI(
    max_indicators=int(max_indicators),
    allow_shorts=allow_shorts,
    risk_cap=risk_cap,
)

st.info(f"PhiAI will optimise **{len(active)}** indicator(s) with up to **{max_iter}** iterations each.", icon="🤖")

# ── Current params table ──────────────────────────────────────────────────────
import pandas as pd

st.subheader("Current Parameters")
rows = []
for name, cfg in active.items():
    rows.append({"Indicator": name, "Params": str(cfg.get("params", {}))})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Run PhiAI ────────────────────────────────────────────────────────────────
if st.button("🤖 Run PhiAI Optimization", type="primary"):
    with st.spinner("PhiAI is searching for optimal parameters…"):
        try:
            optimized, explanation = run_phiai_optimization(
                ohlcv     = ohlcv,
                indicators= active,
                max_iter_per_indicator=max_iter,
                timeframe = timeframe,
            )
            # Update session state for all (active → optimized, inactive → unchanged)
            new_inds = dict(indicators)
            for name, opt_cfg in optimized.items():
                new_inds[name] = opt_cfg
            st.session_state["indicators"] = new_inds

            st.success("PhiAI optimisation complete!")

            # Show changes
            st.subheader("Optimised Parameters")
            opt_rows = []
            for name, cfg in optimized.items():
                opt_rows.append({"Indicator": name, "Best Params": str(cfg.get("params", {}))})
            st.dataframe(pd.DataFrame(opt_rows), use_container_width=True)

            # Explanation
            st.subheader("Explanation")
            st.text(explanation)

        except Exception as exc:
            st.error(f"PhiAI optimisation failed: {exc}")

# ── Skip ──────────────────────────────────────────────────────────────────────
if st.button("Skip — Use Current Params"):
    st.success("Using current parameters without optimisation.")

st.divider()
st.markdown("**Next →** [5 · Backtest Controls](5_Backtest_Controls)")

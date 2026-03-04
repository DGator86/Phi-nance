"""
Page 3 — Signal Blending
=========================

Configure blend method and per-indicator weights.
Stores result in st.session_state["blend_method"] / ["blend_weights"].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phinance.blending import blend_signals, BLEND_METHODS
from phinance.blending.methods import REGIME_INDICATOR_BOOST

st.set_page_config(page_title="Signal Blending | Phi-nance", layout="wide")
st.title("3 · Signal Blending")
st.caption("Choose how to combine your active indicator signals into a single composite.")

indicators = st.session_state.get("indicators", {})
active = {n: v for n, v in indicators.items() if v.get("enabled")}

if not active:
    st.warning("No active indicators. Go back to Step 2 and enable at least one.")
    st.stop()

# ── Method selection ──────────────────────────────────────────────────────────
ai_mode = st.toggle("AI Mode", value=st.session_state.get("blend_method") == "ai_driven")

blend_method = st.radio(
    "Blend Method",
    BLEND_METHODS,
    index=BLEND_METHODS.index(st.session_state.get("blend_method", "weighted_sum")) if st.session_state.get("blend_method", "weighted_sum") in BLEND_METHODS else 0,
    horizontal=True,
)

if ai_mode:
    blend_method = "phiai_chooses"
    c1, c2 = st.columns(2)
    c1.checkbox("Auto-tune blend weights", value=True, key="blend_ai_autotune")
    c2.slider("AI blend window (bars)", 10, 500, 100, 10, key="blend_ai_window")

_METHOD_HELP = {
    "weighted_sum":     "Linear weighted average of all signals.",
    "voting":           "Each indicator votes +1/0/-1; majority wins.",
    "regime_weighted":  "Weights are boosted for indicators that excel in the current regime.",
    "phiai_chooses":    "PhiAI-inspired blending using pre-tuned or auto-derived weights.",
}
st.info(_METHOD_HELP.get(blend_method, ""), icon="ℹ️")

# ── Per-indicator weights ─────────────────────────────────────────────────────
st.subheader("Indicator Weights")
st.caption("Weights are normalised to sum to 1.0 before blending.")

current_weights = st.session_state.get("blend_weights", {})
weights: Dict[str, float] = {}

cols = st.columns(min(len(active), 4))
for idx, name in enumerate(active):
    with cols[idx % len(cols)]:
        default_w = current_weights.get(name, 1.0 / len(active))
        weights[name] = st.slider(
            name,
            min_value=0.0,
            max_value=2.0,
            value=float(default_w),
            step=0.05,
            key=f"wt_{name}",
        )

# Display regime affinity matrix
if blend_method == "regime_weighted":
    with st.expander("Regime–Indicator Affinity Table", expanded=True):
        rows = []
        regimes = ["TREND_UP", "TREND_DN", "RANGE", "BREAKOUT_UP", "BREAKOUT_DN", "HIGHVOL", "LOWVOL"]
        for ind in active:
            boosts = REGIME_INDICATOR_BOOST.get(ind, {})
            row = {"Indicator": ind}
            for r in regimes:
                row[r] = f"{boosts.get(r, 1.0):.1f}×"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Indicator"), use_container_width=True)

# ── Live blend preview ────────────────────────────────────────────────────────
ohlcv = st.session_state.get("ohlcv")
if ohlcv is not None and not ohlcv.empty:
    from phinance.strategies.indicator_catalog import compute_indicator
    import plotly.graph_objects as go

    try:
        sigs = {}
        for name, cfg in active.items():
            sig = compute_indicator(name, ohlcv, cfg.get("params", {}))
            if sig is not None:
                sigs[name] = sig
        if sigs:
            import pandas as _pd
            signals_df = _pd.DataFrame(sigs).reindex(ohlcv.index).ffill().bfill()
            composite = blend_signals(signals_df, weights, blend_method)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=composite.index, y=composite.values,
                name="Composite Signal",
                fill="tozeroy",
                line=dict(color="#6366f1"),
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="#475569")
            fig.update_layout(
                title="Composite Signal Preview",
                height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_range=[-1.2, 1.2],
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Preview failed: {exc}")

# ── Save ──────────────────────────────────────────────────────────────────────
if st.button("Save Blend Configuration", type="primary"):
    st.session_state["blend_method"]  = blend_method
    st.session_state["blend_weights"] = weights
    st.success(f"Blend method: **{blend_method}** with {len(weights)} weighted indicators.")

st.divider()
st.markdown("**Next →** [4 · PhiAI Optimization](4_PhiAI_Optimization)")

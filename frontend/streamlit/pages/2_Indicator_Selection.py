"""
Page 2 — Indicator Selection
=============================

Enable / configure individual indicator strategies.
Stores result in st.session_state["indicators"].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phinance.strategies import list_indicators, INDICATOR_CATALOG
from phinance.strategies.params import get_default_params

st.set_page_config(page_title="Indicator Selection | Phi-nance", layout="wide")
st.title("2 · Indicator Selection")
st.caption("Enable indicators and set their starting parameters.")

timeframe = st.session_state.get("timeframe", "1D")
current   = st.session_state.get("indicators", {})
all_names = list_indicators()

updated: Dict[str, Dict[str, Any]] = {}

cols = st.columns(2)
for idx, name in enumerate(all_names):
    col = cols[idx % 2]
    with col:
        with st.expander(f"**{name}**", expanded=bool(current.get(name, {}).get("enabled", False))):
            enabled = st.checkbox(
                "Enable",
                value=current.get(name, {}).get("enabled", False),
                key=f"ind_enabled_{name}",
            )
            defaults = get_default_params(name, timeframe)
            indicator = INDICATOR_CATALOG[name]
            params: Dict[str, Any] = {}

            if defaults:
                st.caption("Parameters")
                cfg_params = current.get(name, {}).get("params", defaults)
                for param, default_val in defaults.items():
                    if isinstance(default_val, float):
                        val = st.number_input(
                            param,
                            value=float(cfg_params.get(param, default_val)),
                            step=0.1,
                            format="%.2f",
                            key=f"ind_{name}_{param}",
                        )
                    else:
                        val = st.number_input(
                            param,
                            value=int(cfg_params.get(param, default_val)),
                            step=1,
                            key=f"ind_{name}_{param}",
                        )
                    params[param] = val
            else:
                st.caption("No configurable parameters.")

            updated[name] = {
                "enabled":   enabled,
                "auto_tune": current.get(name, {}).get("auto_tune", True),
                "params":    params or defaults,
            }

if st.button("Save Indicator Configuration", type="primary"):
    enabled_count = sum(1 for v in updated.values() if v["enabled"])
    st.session_state["indicators"] = updated
    st.success(f"Saved {enabled_count} active indicator(s).")

# ── Preview active indicators ─────────────────────────────────────────────────
active = {n: v for n, v in updated.items() if v.get("enabled")}
if active:
    st.subheader("Active Indicators")
    rows = [{"Indicator": n, "Params": str(v["params"])} for n, v in active.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Live signal preview on current OHLCV
    ohlcv = st.session_state.get("ohlcv")
    if ohlcv is not None and not ohlcv.empty and st.checkbox("Preview signals on data"):
        from phinance.strategies.indicator_catalog import compute_indicator
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["close"], name="Close", line=dict(color="#94a3b8", width=1)))
        for ind_name, ind_cfg in active.items():
            try:
                sig = compute_indicator(ind_name, ohlcv, ind_cfg["params"])
                if sig is not None:
                    # Scale signal to price range for overlay
                    price_range = float(ohlcv["close"].max() - ohlcv["close"].min())
                    mid = float(ohlcv["close"].mean())
                    overlay = mid + sig * price_range * 0.4
                    fig.add_trace(go.Scatter(x=ohlcv.index, y=overlay, name=ind_name, opacity=0.7))
            except Exception as exc:
                st.warning(f"{ind_name}: {exc}")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enable at least one indicator above.")

st.divider()
st.markdown("**Next →** [3 · Blending](3_Blending)")

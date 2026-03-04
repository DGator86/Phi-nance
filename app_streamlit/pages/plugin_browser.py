"""
app_streamlit/pages/plugin_browser.py
=======================================

Streamlit page: Plugin Browser
Allows users to browse registered third-party indicators and data vendors,
register new plugins via upload or paste, and inspect metadata.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from phinance.plugins.registry import (
    get_registry,
    list_plugins,
    load_plugin_directory,
    reset_registry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(text: str, colour: str = "#a855f7") -> str:
    return (
        f'<span style="background:{colour}22;color:{colour};border:1px solid {colour}44;'
        f'padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;">'
        f"{text}</span>"
    )


def render_plugin_browser():
    """Render the Plugin Browser page."""
    st.title("🔌 Plugin Browser")
    st.caption("Browse, inspect, and load third-party indicators and data vendors.")

    plugins = list_plugins()
    indicators = plugins.get("indicators", [])
    vendors    = plugins.get("vendors", [])

    # ── Summary cards ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Registered Indicators", len(indicators))
    with col2:
        st.metric("Registered Vendors", len(vendors))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ind, tab_ven, tab_load = st.tabs(["📈 Indicators", "📦 Vendors", "⬆️ Load Plugin"])

    # ── Indicator tab ─────────────────────────────────────────────────────────
    with tab_ind:
        if not indicators:
            st.info("No plugin indicators registered yet. Use the **Load Plugin** tab to add one.")
        else:
            st.subheader(f"Plugin Indicators ({len(indicators)})")
            reg = get_registry()
            rows = []
            for name in sorted(indicators):
                ind = reg.get_indicator(name)
                meta = reg.get_metadata("indicators", name)
                rows.append({
                    "Name":        name,
                    "Class":       type(ind).__name__ if ind else "—",
                    "Version":     meta.get("version", "—"),
                    "Author":      meta.get("author", "—"),
                    "Description": meta.get("description", "—"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Detail view
            selected = st.selectbox("Inspect indicator", ["— select —"] + sorted(indicators))
            if selected != "— select —":
                ind = reg.get_indicator(selected)
                if ind:
                    st.markdown(f"**Class**: `{type(ind).__name__}`")
                    with st.expander("Default Parameters"):
                        dp = getattr(ind, "default_params", {})
                        if dp:
                            st.json(dp)
                        else:
                            st.write("No default parameters.")
                    with st.expander("Parameter Grid"):
                        pg = getattr(ind, "param_grid", {})
                        if pg:
                            st.json(pg)
                        else:
                            st.write("No parameter grid.")
                    with st.expander("Metadata"):
                        meta = reg.get_metadata("indicators", selected)
                        if meta:
                            st.json(meta)
                        else:
                            st.write("No metadata registered.")

    # ── Vendor tab ────────────────────────────────────────────────────────────
    with tab_ven:
        if not vendors:
            st.info("No plugin vendors registered yet. Use the **Load Plugin** tab to add one.")
        else:
            st.subheader(f"Plugin Vendors ({len(vendors)})")
            reg = get_registry()
            rows = []
            for name in sorted(vendors):
                vendor = reg.get_vendor(name)
                meta   = reg.get_metadata("vendors", name)
                rows.append({
                    "Name":        name,
                    "Callable":    getattr(vendor, "__name__", str(vendor)),
                    "Version":     meta.get("version", "—"),
                    "Author":      meta.get("author", "—"),
                    "Description": meta.get("description", "—"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Load plugin tab ───────────────────────────────────────────────────────
    with tab_load:
        st.subheader("Load Plugin from File")
        st.write(
            "Upload a Python file that uses `@register_indicator_plugin` or "
            "`@register_vendor_plugin` decorators. The file will be executed and "
            "plugins auto-discovered."
        )

        uploaded = st.file_uploader("Upload plugin (.py)", type=["py"])
        if uploaded is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                plugin_path = os.path.join(tmpdir, uploaded.name)
                with open(plugin_path, "wb") as f:
                    f.write(uploaded.read())
                try:
                    load_plugin_directory(tmpdir)
                    st.success(f"✅ Plugin `{uploaded.name}` loaded successfully!")
                    new_plugins = list_plugins()
                    st.json(new_plugins)
                except Exception as exc:
                    st.error(f"❌ Failed to load plugin: {exc}")

        st.divider()
        st.subheader("Load Plugin from Directory Path")
        dir_path = st.text_input("Directory path (sandbox-visible)", placeholder="/path/to/plugins")
        if st.button("🔍 Scan & Load") and dir_path:
            if os.path.isdir(dir_path):
                try:
                    load_plugin_directory(dir_path)
                    st.success(f"✅ Plugins from `{dir_path}` loaded.")
                    st.json(list_plugins())
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")
            else:
                st.warning(f"Directory not found: `{dir_path}`")

        st.divider()
        with st.expander("📋 Plugin Registration Template"):
            st.code('''
from phinance.strategies.base import BaseIndicator
from phinance.plugins.registry import register_indicator_plugin
import pandas as pd

@register_indicator_plugin("My Custom RSI")
class MyCustomRSI(BaseIndicator):
    """Custom RSI variant with adaptive smoothing."""

    @property
    def default_params(self):
        return {"period": 14}

    @property
    def param_grid(self):
        return {"period": [7, 10, 14, 20]}

    def compute(self, ohlcv: pd.DataFrame, period: int = 14, **_) -> pd.Series:
        close  = ohlcv["close"]
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, 1e-10)
        rsi    = 100 - 100 / (1 + rs)
        signal = (rsi - 50) / 50          # scale to [-1, +1]
        return signal.fillna(0).rename("My Custom RSI")
''', language="python")


# ── Standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    render_plugin_browser()

#!/usr/bin/env python3
"""
Phi-nance Live Workbench — Multi-Page Streamlit App
=====================================================

Entry point for the Phi-nance Streamlit application.

Run:
    streamlit run frontend/streamlit/live_workbench.py

This file bootstraps the app, injects global CSS, configures session
state, and renders a landing page that routes to the workflow pages.
Each workflow step is a separate page under ``frontend/streamlit/pages/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("IS_BACKTESTING", "True")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phi-nance Workbench",
    page_icon="Φ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ─────────────────────────────────────────────────────────────
_CSS_PATH = _ROOT / ".streamlit" / "styles.css"

def _inject_css() -> None:
    css = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_inject_css()

# ── Session state initialisation ─────────────────────────────────────────────
_DEFAULTS = {
    "step": 1,
    "ohlcv": None,
    "vendor": "alphavantage",
    "symbol": "SPY",
    "timeframe": "1D",
    "start_date": "2022-01-01",
    "end_date": "2024-12-31",
    "indicators": {},
    "blend_method": "weighted_sum",
    "blend_weights": {},
    "phiai_enabled": False,
    "initial_capital": 100_000.0,
    "signal_threshold": 0.15,
    "trading_mode": "equities",
    "backtest_result": None,
    "run_id": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Landing page ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 2rem 0 1rem">
        <h1 style="font-size:3rem; font-weight:800; letter-spacing:-1px;">
            Φ Phi-nance
        </h1>
        <p style="font-size:1.2rem; color:#94a3b8;">
            Open-Source Quantitative Research Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        ### 🚀 Workflow
        Follow the **6-step pipeline** in the sidebar:
        1. Build your dataset
        2. Select indicators
        3. Blend signals
        4. PhiAI optimisation
        5. Run backtest
        6. Analyse results
        """
    )
with col2:
    st.markdown(
        """
        ### ⚡ Features
        - **3 data vendors**: yfinance, Alpha Vantage, Binance
        - **8 indicators** with auto-tuning
        - **4 blend methods** including regime-weighted
        - **Options backtesting** with Black-Scholes Greeks
        - **Phibot** AI post-run reviewer
        """
    )
with col3:
    st.markdown(
        """
        ### 📂 Documentation
        - [Architecture](docs/architecture.md)
        - [API Reference](docs/api_reference.md)
        - [Contributing](CONTRIBUTING.md)
        - [GitHub](https://github.com/DGator86/Phi-nance)
        """
    )

st.info(
    "👈 Use the sidebar to navigate through the workflow steps, "
    "or jump directly to any page.",
    icon="ℹ️",
)

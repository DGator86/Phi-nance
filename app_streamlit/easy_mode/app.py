"""Phi-nance easy mode — three screens, no tuning required."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from app_streamlit.easy_mode.backtest_screen import render_auto_backtest
from app_streamlit.easy_mode.constants import UNIVERSE
from app_streamlit.easy_mode.overview import render_overview
from app_streamlit.easy_mode.ticker_screen import render_ticker_spotlight
from phi.config import settings

os.environ.setdefault("IS_BACKTESTING", "True")
load_dotenv()
settings.create_dirs()


def _inject_theme() -> None:
    st.markdown(
        """
        <style>
          .phinance-hero {
            font-size: 1.85rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
            background: linear-gradient(105deg, #38bdf8 0%, #a78bfa 45%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
          }
          [data-testid="stSidebar"] {
            background: linear-gradient(185deg, #0f172a 0%, #1e1b4b 100%);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_easy_app() -> None:
    st.set_page_config(
        page_title="Phi-nance",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()

    st.sidebar.markdown("### ◈ Phi-nance")
    st.sidebar.caption("Simple trading desk")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Automatic backtest", "Ticker spotlight"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Universe:** set `PHINANCE_UNIVERSE` (comma-separated tickers). "
        f"Now watching: **{', '.join(UNIVERSE)}**."
    )
    st.sidebar.caption(
        "Advanced controls: `streamlit run app_streamlit/expert_workbench.py`"
    )

    if page == "Overview":
        render_overview()
    elif page == "Automatic backtest":
        render_auto_backtest()
    else:
        render_ticker_spotlight()


def main() -> None:
    run_easy_app()


if __name__ == "__main__":
    main()

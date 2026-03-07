"""Main entry point for the modular Streamlit live backtest workbench."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from app_streamlit.state import AppState, init_session_state, reset_state
from app_streamlit.ui_components import (
    render_config_panel,
    render_error,
    render_form_errors,
    render_loaded_config_summary,
    render_results,
    render_run_history,
)
from app_streamlit.ui_handlers import handle_load_run, handle_run_backtest
from phi.config import settings

os.environ.setdefault("IS_BACKTESTING", "True")
load_dotenv()


def main() -> None:
    """Render app layout and dispatch actions based on state machine value."""
    st.set_page_config(page_title="Phi-nance Live Workbench", layout="wide")
    init_session_state()

    st.title("Live Backtest Workbench")
    render_loaded_config_summary(st.session_state.get("config"))

    with st.sidebar:
        st.header("Configuration")
        payload, run_clicked = render_config_panel()
        if st.button("Reset workbench"):
            reset_state()
            st.rerun()

        selected_run_id = render_run_history()
        if selected_run_id:
            handle_load_run(selected_run_id)
            st.rerun()

    if run_clicked:
        handle_run_backtest(payload)

    if st.session_state.form_errors:
        render_form_errors(st.session_state.form_errors)

    state = st.session_state.app_state
    if state == AppState.IDLE:
        st.info("Configure your backtest in the sidebar, then click Run backtest.")
    elif state == AppState.CONFIGURING:
        st.info("Configuration validated. Click Run backtest when ready.")
    elif state == AppState.RUNNING:
        st.info("Running backtest...")
    elif state == AppState.RESULTS:
        render_results(st.session_state.results or {})
    elif state == AppState.ERROR:
        render_error(st.session_state.error, st.session_state.error_debug, settings.DEBUG)


if __name__ == "__main__":
    main()

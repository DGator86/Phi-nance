"""Session state machine and mutation helpers for Streamlit workbench."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import streamlit as st


class AppState(Enum):
    """Lifecycle states for workbench rendering and action flow."""

    IDLE = "idle"
    CONFIGURING = "configuring"
    RUNNING = "running"
    RESULTS = "results"
    ERROR = "error"


DEFAULT_STATE: dict[str, Any] = {
    "app_state": AppState.IDLE,
    "config": None,
    "results": None,
    "error": None,
    "error_debug": None,
    "loaded_data": None,
    "last_run_id": None,
    "form_errors": [],
}


CONFIG_INPUT_KEYS = [
    "symbol",
    "start_date",
    "end_date",
    "timeframe",
    "vendor",
    "initial_capital",
    "trading_mode",
    "selected_indicators",
    "blend_method",
    "option_type",
    "option_strike",
    "option_expiry",
    "option_iv",
    "option_rate",
    "option_qty",
]


def init_session_state() -> None:
    """Ensure required session state keys are initialized once."""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def transition_to(new_state: AppState, error: Optional[str] = None, debug: Optional[str] = None) -> None:
    """Transition app state and optionally attach user + debug error metadata."""
    st.session_state.app_state = new_state
    if error is not None:
        st.session_state.error = error
    if debug is not None:
        st.session_state.error_debug = debug


def set_config(config: dict[str, Any]) -> None:
    """Persist validated run config and move to configuring state."""
    st.session_state.config = config
    transition_to(AppState.CONFIGURING)


def set_results(results: dict[str, Any]) -> None:
    """Persist run results and move to results state."""
    st.session_state.results = results
    st.session_state.error = None
    st.session_state.error_debug = None
    transition_to(AppState.RESULTS)


def set_error(message: str, debug: Optional[str] = None) -> None:
    """Store the current failure and move to error state."""
    transition_to(AppState.ERROR, error=message, debug=debug)


def set_form_errors(errors: list[str]) -> None:
    """Store inline form validation errors."""
    st.session_state.form_errors = errors


def reset_state() -> None:
    """Clear all workbench state and input widgets for a clean restart."""
    for key in CONFIG_INPUT_KEYS:
        st.session_state.pop(key, None)
    for key in list(DEFAULT_STATE):
        st.session_state[key] = DEFAULT_STATE[key]

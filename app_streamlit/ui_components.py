"""Reusable Streamlit UI building blocks for the live workbench."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app_streamlit.config import (
    BLEND_METHOD_OPTIONS,
    DEFAULT_END_DATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_TRADING_MODE,
    DEFAULT_VENDOR,
    INDICATOR_SPECS,
    TIMEFRAME_OPTIONS,
    TRADING_MODE_OPTIONS,
    VENDOR_OPTIONS,
)
from phi.config import settings
from phi.run_config import RunConfig, RunHistory


def render_indicator_selector(selected_names: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Render indicator multiselect and per-indicator parameter controls."""
    selected_names = st.multiselect(
        "Indicators",
        options=list(INDICATOR_SPECS.keys()),
        default=selected_names,
        key="selected_indicators",
    )
    indicators: dict[str, dict[str, Any]] = {}
    blend_weights: dict[str, float] = {}

    for name in selected_names:
        spec = INDICATOR_SPECS[name]
        with st.expander(name, expanded=False):
            st.caption(spec.description)
            params: dict[str, Any] = {}
            for param, (min_v, max_v, default_v, step) in spec.params.items():
                params[param] = st.number_input(
                    param,
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=float(default_v),
                    step=float(step),
                    key=f"{name}_{param}",
                )
            indicators[name] = {"enabled": True, "params": params}

    if selected_names:
        st.caption("Blend weights (used for weighted_sum mode)")
        default_weight = round(1.0 / len(selected_names), 4)
        for name in selected_names:
            blend_weights[name] = st.slider(
                f"Weight: {name}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01,
                key=f"blend_weight_{name}",
            )
    return indicators, blend_weights


def render_date_picker(default_start: date, default_end: date) -> tuple[date, date]:
    """Render start/end date controls and return selected values."""
    c1, c2 = st.columns(2)
    start = c1.date_input("Start date", value=default_start, key="start_date")
    end = c2.date_input("End date", value=default_end, key="end_date")
    return start, end


def render_config_panel() -> tuple[dict[str, Any], bool]:
    """Render sidebar configuration form and return payload + run click state."""
    with st.form("run_config_form", clear_on_submit=False):
        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL, key="symbol").strip().upper()
        timeframe = st.selectbox("Timeframe", TIMEFRAME_OPTIONS, index=TIMEFRAME_OPTIONS.index(DEFAULT_TIMEFRAME), key="timeframe")
        vendor = st.selectbox("Data vendor", VENDOR_OPTIONS, index=VENDOR_OPTIONS.index(DEFAULT_VENDOR), key="vendor")
        start_date, end_date = render_date_picker(DEFAULT_START_DATE, DEFAULT_END_DATE)
        initial_capital = st.number_input("Initial capital", min_value=1000.0, value=float(DEFAULT_INITIAL_CAPITAL), step=1000.0, key="initial_capital")
        trading_mode = st.selectbox("Trading mode", TRADING_MODE_OPTIONS, index=TRADING_MODE_OPTIONS.index(DEFAULT_TRADING_MODE), key="trading_mode")
        blend_method = st.selectbox("Blend method", BLEND_METHOD_OPTIONS, key="blend_method")
        indicators, blend_weights = render_indicator_selector(st.session_state.get("selected_indicators", []))

        option_type = option_strike = option_expiry = option_iv = option_rate = option_qty = None
        if trading_mode == "options":
            st.markdown("#### Options setup")
            option_type = st.selectbox("Option type", ["call", "put"], key="option_type")
            option_strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0, key="option_strike")
            option_expiry = st.date_input("Expiry", value=end_date, key="option_expiry")
            option_iv = st.number_input("Implied volatility", min_value=0.01, max_value=5.0, value=0.3, step=0.01, key="option_iv")
            option_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.5, value=0.02, step=0.005, key="option_rate")
            option_qty = st.number_input("Contracts", min_value=1, max_value=1000, value=1, step=1, key="option_qty")

        run_clicked = st.form_submit_button("Run backtest", type="primary")

    payload = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "timeframe": timeframe,
        "vendor": vendor,
        "initial_capital": initial_capital,
        "trading_mode": trading_mode,
        "indicators": indicators,
        "blend_method": blend_method,
        "blend_weights": blend_weights,
        "option_type": option_type,
        "option_strike": option_strike,
        "option_expiry": option_expiry,
        "option_iv": option_iv,
        "option_rate": option_rate,
        "option_qty": option_qty,
    }
    return payload, run_clicked


def render_form_errors(errors: list[str]) -> None:
    """Render inline form validation errors in sidebar/main areas."""
    for err in errors:
        st.warning(err)


def render_run_history(runs_dir: Path | None = None) -> str | None:
    """Display recent run IDs and return selected run id for loading."""
    history = RunHistory(runs_dir or settings.RUNS_DIR)
    runs = history.list_runs()
    if not runs:
        st.caption("No saved runs yet.")
        return None

    options = [r["run_id"] for r in runs]
    selected = st.selectbox("Load historical run", options=options, key="history_run_id")
    return selected if st.button("Load selected run") else None


def render_results(results: dict[str, Any]) -> None:
    """Display top-level metrics, equity curve, and trade details."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{results.get('total_return', 0) * 100:.2f}%")
    c2.metric("CAGR", f"{results.get('cagr', 0) * 100:.2f}%")
    c3.metric("Sharpe", f"{results.get('sharpe', 0):.2f}")
    c4.metric("Max Drawdown", f"{results.get('max_drawdown', 0) * 100:.2f}%")

    pv = results.get("portfolio_value", [])
    if pv:
        st.line_chart(pd.Series(pv, name="Portfolio Value"))

    trades = results.get("trades", [])
    if trades:
        st.subheader("Trades")
        st.dataframe(pd.DataFrame(trades), use_container_width=True)

    run_id = results.get("run_id")
    if run_id:
        st.caption(f"Saved run id: {run_id}")


def render_error(error_message: str | None, debug_details: str | None, debug_enabled: bool) -> None:
    """Render friendly error and optional debug traceback expander."""
    st.error(error_message or "Unexpected error occurred.")
    if debug_enabled and debug_details:
        with st.expander("Debug details"):
            st.code(debug_details, language="text")


def render_loaded_config_summary(config_payload: dict[str, Any] | None) -> None:
    """Render a compact summary for currently active configuration."""
    if not config_payload:
        return
    try:
        cfg = RunConfig.model_validate(config_payload)
    except Exception:  # noqa: BLE001
        return
    st.caption(
        f"{cfg.symbols[0]} | {cfg.start_date} → {cfg.end_date} | {cfg.timeframe} | {cfg.trading_mode}"
    )

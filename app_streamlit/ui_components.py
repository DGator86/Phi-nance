"""Reusable Streamlit UI building blocks for the live workbench."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError as PydanticValidationError

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
from phi.regime import list_saved_detectors, load_detector
from phi.logging import get_logger
from phi.run_config import RunConfig, RunHistory

logger = get_logger(__name__)


def render_options_playbook_sidebar() -> None:
    """Show composite regime × vol playbook rows next to options fields."""
    from phi.options.regime_playbook import get_default_options_regime_playbook

    pb = get_default_options_regime_playbook()
    with st.expander("Regime playbook (options cheat sheet)", expanded=False):
        st.caption(
            "Each row ties a **trend × volatility** label to approved structures and a risk envelope. "
            "Enable **regime-aware blending** below and select a saved model to tag options backtests "
            "and see PnL by regime in results."
        )
        rk = st.selectbox(
            "Inspect row",
            options=sorted(pb.regimes.keys()),
            key="options_playbook_regime_key",
        )
        entry = pb.regimes[rk]
        st.markdown(f"**{entry.display_name}** — {entry.summary}")
        st.write(
            f"DTE **{entry.dte_days_min}–{entry.dte_days_max}** trading days | "
            f"Δ **{entry.delta_band[0]:.2f}–{entry.delta_band[1]:.2f}** | "
            f"max risk guideline **~{entry.max_risk_pct_portfolio * 100:.1f}%** of portfolio"
        )
        structs = entry.allowed_structures
        st.caption("Structures: " + ", ".join(structs[:10]) + (" …" if len(structs) > 10 else ""))
        if rk.startswith("BULL"):
            st.caption("Directional hint: bullish structures / calls more natural in this row.")
        elif rk.startswith("BEAR"):
            st.caption("Directional hint: bearish structures / puts more natural in this row.")
        else:
            st.caption("Directional hint: neutral / range structures often fit ranging regimes.")


def render_indicator_selector(selected_names: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Render indicator multiselect and per-indicator parameter controls."""
    categories: dict[str, list[str]] = {}
    for indicator_name, spec in INDICATOR_SPECS.items():
        categories.setdefault(spec.category, []).append(indicator_name)

    ordered_options: list[str] = []
    for category in sorted(categories.keys()):
        ordered_options.extend(sorted(categories[category]))

    selected_names = st.multiselect(
        "Indicators",
        options=ordered_options,
        default=selected_names,
        key="selected_indicators",
    )

    category_labels = []
    for category in sorted(categories.keys()):
        names = ", ".join(sorted(categories[category]))
        category_labels.append(f"**{category}:** {names}")
    st.caption(" | ".join(category_labels))
    indicators: dict[str, dict[str, Any]] = {}
    blend_weights: dict[str, float] = {}

    for name in selected_names:
        spec = INDICATOR_SPECS[name]
        with st.expander(name, expanded=False):
            st.caption(spec.description)
            params: dict[str, Any] = {}
            for param, param_spec in spec.params.items():
                if isinstance(param_spec, tuple):
                    min_v, max_v, default_v, step = param_spec
                    params[param] = st.number_input(
                        param,
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(default_v),
                        step=float(step),
                        key=f"{name}_{param}",
                    )
                elif isinstance(param_spec, dict) and param_spec.get("type") == "select":
                    options = list(param_spec.get("options", []))
                    default_value = param_spec.get("default", options[0] if options else "")
                    default_idx = options.index(default_value) if default_value in options else 0
                    params[param] = st.selectbox(
                        param,
                        options=options,
                        index=default_idx,
                        key=f"{name}_{param}",
                    )
                else:
                    logger.warning("Unsupported parameter spec for %s/%s: %s", name, param, param_spec)
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
        symbols_raw = st.text_input("Symbols (comma-separated)", value=DEFAULT_SYMBOL, key="symbols").strip().upper()
        symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        symbol = symbols[0] if symbols else DEFAULT_SYMBOL
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
            st.caption(
                "Vendor must be **unusual_whales**. Strike, expiry, and IV are taken from the "
                "Unusual Whales chain (ATM row with live Greeks). Flow alerts are summarized on the results screen."
            )
            option_type = st.selectbox("Option type", ["call", "put"], key="option_type")
            option_strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0, key="option_strike")
            option_expiry = st.date_input("Expiry", value=end_date, key="option_expiry")
            option_iv = st.number_input("Implied volatility", min_value=0.01, max_value=5.0, value=0.3, step=0.01, key="option_iv")
            option_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.5, value=0.02, step=0.005, key="option_rate")
            option_qty = st.number_input("Contracts", min_value=1, max_value=1000, value=1, step=1, key="option_qty")
            render_options_playbook_sidebar()

        portfolio_payload = render_portfolio_panel(symbols)
        run_clicked = st.form_submit_button("Run backtest", type="primary")

    regime_payload = render_regime_detection_panel(indicators)

    payload = {
        "symbol": symbol,
        "symbols": symbols,
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
        **portfolio_payload,
        **regime_payload,
    }
    return payload, run_clicked




def render_portfolio_panel(symbols: list[str]) -> dict[str, Any]:
    """Render portfolio allocation + rebalance controls."""
    st.markdown("#### Portfolio Configuration")
    allocation_strategy = st.selectbox(
        "Allocation strategy",
        options=["equal_weight", "fixed_weight", "signal_weighted", "risk_parity"],
        key="allocation_strategy",
    )

    fixed_weights: dict[str, float] = {}
    if allocation_strategy == "fixed_weight":
        st.caption("Fixed weights (should sum to 1.0)")
        default_weight = 1.0 / max(len(symbols), 1)
        for sym in symbols:
            fixed_weights[sym] = st.number_input(
                f"Weight {sym}",
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.01,
                key=f"fixed_weight_{sym}",
            )

    rebalance_frequency = st.selectbox(
        "Rebalance frequency",
        options=["none", "D", "W", "M", "Q", 5, 20],
        key="rebalance_frequency",
    )
    rebalance_threshold_enabled = st.checkbox("Enable threshold rebalance", value=False, key="rebalance_threshold_enabled")
    rebalance_threshold = st.slider("Threshold", min_value=0.0, max_value=0.5, value=0.05, step=0.01, key="rebalance_threshold")

    allocation_params: dict[str, Any] = {}
    if fixed_weights:
        allocation_params["weights"] = fixed_weights

    return {
        "allocation_strategy": allocation_strategy,
        "allocation_params": allocation_params,
        "rebalance_frequency": rebalance_frequency,
        "rebalance_threshold": rebalance_threshold if rebalance_threshold_enabled else None,
    }

def render_regime_detection_panel(indicators: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Render sidebar controls for regime-aware blending configuration."""
    with st.expander("Regime-Aware Blending", expanded=False):
        regime_enabled = st.checkbox("Enable regime-aware blending", value=False, key="regime_enabled")
        method_label = st.selectbox(
            "Method",
            options=["HMM", "Clustering (KMeans)", "GMM", "Deep Learning (LSTM)", "Deep Learning (Transformer)"],
            key="regime_method",
        )
        n_states = st.slider("Number of regimes", min_value=2, max_value=8, value=3, key="regime_n_states")
        window = st.slider("Feature window", min_value=5, max_value=60, value=20, key="regime_window")
        train_clicked = st.button("Train on selected range", key="train_regime_model")

        deep_seq_length = st.slider("Sequence length", min_value=5, max_value=120, value=20, key="regime_deep_seq_length")
        deep_hidden_size = st.slider("Hidden size", min_value=16, max_value=256, value=64, step=16, key="regime_deep_hidden_size")
        deep_num_layers = st.slider("Layers", min_value=1, max_value=6, value=2, key="regime_deep_num_layers")
        deep_epochs = st.slider("Epochs", min_value=1, max_value=200, value=50, key="regime_deep_epochs")
        deep_batch_size = st.slider("Batch size", min_value=8, max_value=256, value=32, step=8, key="regime_deep_batch_size")
        deep_lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f", key="regime_deep_lr")

        uploaded_deep_model = None
        if method_label.startswith("Deep Learning"):
            st.info("Deep model training is done via `python -m phi.regime.train_deep`; use the uploader to load a trained model.")
            uploaded_deep_model = st.file_uploader(
                "Upload pre-trained deep detector (.pkl)",
                type=["pkl"],
                key="regime_deep_model_upload",
            )

        refresh_models = st.button("Refresh detector list", key="refresh_regime_models")
        if refresh_models or "regime_available_models" not in st.session_state:
            st.session_state.regime_available_models = list_saved_detectors(settings.REGIME_MODELS_DIR)

        available = st.session_state.get("regime_available_models", [])
        label_to_model: dict[str, dict[str, Any]] = {}
        model_labels: list[str] = []
        for model in available:
            metadata = model.get("metadata", {})
            detector_class = metadata.get("detector_class", "Unknown")
            period = metadata.get("training_period", {})
            start = period.get("start", "?")
            end = period.get("end", "?")
            label = f"{model['name']} | {detector_class} | {start} → {end}"
            model_labels.append(label)
            label_to_model[label] = model

        selected_label = st.selectbox(
            "Saved detector",
            options=["(none)", *model_labels],
            key="regime_selected_model_label",
        )

        selected_model_path = None
        selected_model_meta: dict[str, Any] = {}
        if selected_label != "(none)":
            selected = label_to_model[selected_label]
            selected_model_path = selected.get("path")
            selected_model_meta = selected.get("metadata", {})
            st.caption(f"Selected model: `{selected_model_path}`")
            if selected_model_meta:
                st.json(selected_model_meta)

        detect_on_the_fly = st.checkbox(
            "Use detector on-the-fly during backtest",
            value=True,
            key="regime_detect_on_the_fly",
        )
        use_precomputed = st.checkbox(
            "Use pre-computed regime series from session (if available)",
            value=False,
            key="regime_use_precomputed",
        )

        selected_model_source = "saved"
        if uploaded_deep_model is not None:
            uploads_dir = settings.REGIME_MODELS_DIR / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            upload_path = uploads_dir / uploaded_deep_model.name
            upload_path.write_bytes(uploaded_deep_model.getbuffer())
            selected_model_path = str(upload_path)
            selected_model_source = "uploaded"
            st.caption(f"Uploaded model ready: `{selected_model_path}`")

        regime_label_map: dict[str, str] = {}
        regime_boost_matrix: dict[str, dict[str, float]] = {}
        if regime_enabled and selected_model_path:
            detector = load_detector(selected_model_path)
            metadata = selected_model_meta or getattr(detector, "metadata", {})
            params = metadata.get("params", {})
            regime_count = int(params.get("n_states") or params.get("n_clusters") or n_states)
            default_labels = [f"state_{i}" for i in range(regime_count)]
            enabled_indicators = [name for name, cfg in indicators.items() if cfg.get("enabled")]
            if not enabled_indicators:
                st.caption("Enable indicators to configure regime boosts.")
            for raw_label in default_labels:
                friendly = st.text_input(
                    f"Friendly name for {raw_label}",
                    value=str(st.session_state.get(f"regime_friendly_{raw_label}", raw_label)),
                    key=f"regime_friendly_{raw_label}",
                )
                regime_label_map[raw_label] = friendly.strip() or raw_label

            if enabled_indicators:
                df = pd.DataFrame(index=enabled_indicators)
                for raw_label in default_labels:
                    col = regime_label_map.get(raw_label, raw_label)
                    df[col] = 1.0
                edited = st.data_editor(
                    df,
                    num_rows="fixed",
                    use_container_width=True,
                    key="regime_boost_editor",
                )
                for raw_label in default_labels:
                    friendly = regime_label_map.get(raw_label, raw_label)
                    regime_boost_matrix[raw_label] = {
                        indicator: float(edited.loc[indicator, friendly])
                        for indicator in enabled_indicators
                    }

    return {
        "regime_enabled": regime_enabled,
        "regime_method": method_label,
        "regime_n_states": int(n_states),
        "regime_window": int(window),
        "regime_train_clicked": bool(train_clicked),
        "regime_selected_model_path": selected_model_path,
        "regime_detect_on_the_fly": bool(detect_on_the_fly),
        "regime_use_precomputed": bool(use_precomputed),
        "regime_model_source": selected_model_source,
        "regime_deep_seq_length": int(deep_seq_length),
        "regime_deep_hidden_size": int(deep_hidden_size),
        "regime_deep_num_layers": int(deep_num_layers),
        "regime_deep_epochs": int(deep_epochs),
        "regime_deep_batch_size": int(deep_batch_size),
        "regime_deep_lr": float(deep_lr),
        "regime_label_map": regime_label_map,
        "regime_boost_matrix": regime_boost_matrix,
    }


def render_regime_chart(price_data: pd.DataFrame, regime_series: pd.Series) -> None:
    """Render price with regime-colored overlays."""
    if price_data.empty or regime_series.empty:
        return

    df = price_data.copy()
    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        return

    original = regime_series.reindex(df.index)
    aligned = original.ffill().bfill()

    filled_count = int(aligned.isna().sum())
    if filled_count > 0:
        st.warning(f"Regime series could not be fully aligned; {filled_count} bars have no regime.")
    else:
        filled_rows = int((original != aligned).fillna(False).sum())
        total_rows = len(df)
        if total_rows > 0 and filled_rows > total_rows * 0.05:
            st.warning(
                f"Large regime alignment: {filled_rows} out of {total_rows} bars "
                f"({filled_rows/total_rows:.1%}) were filled to match price data. "
                "Check that the training date range matches the backtest range."
            )

    aligned = aligned.astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[close_col], mode="lines", name="Close", line={"color": "#4ea1ff"}))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for idx, regime in enumerate(sorted(aligned.unique())):
        mask = aligned == regime
        fig.add_trace(
            go.Scatter(
                x=df.index[mask],
                y=df.loc[mask, close_col],
                mode="markers",
                marker={"size": 5, "color": palette[idx % len(palette)]},
                name=regime,
                opacity=0.6,
            )
        )

    fig.update_layout(title="Detected Regimes", xaxis_title="Date", yaxis_title="Price", height=420)
    st.plotly_chart(fig, use_container_width=True)


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

    rae = results.get("regime_at_entry")
    if rae:
        st.info(f"**Regime at option entry:** `{rae}`")

    mbr = results.get("metrics_by_regime")
    if isinstance(mbr, dict) and mbr:
        st.subheader("Attributed daily PnL by regime")
        st.caption("Day-over-day portfolio change summed by the regime label on that bar (requires regime model + options run).")
        st.dataframe(
            pd.DataFrame.from_dict(mbr, orient="index"),
            use_container_width=True,
        )

    opb = results.get("options_regime_playbook")
    if isinstance(opb, dict) and opb.get("transitions"):
        with st.expander("Regime transition map (what to watch)"):
            for row in opb["transitions"]:
                f = row.get("from") or row.get("from_regime", "")
                t = row.get("to") or row.get("to_regime", "")
                st.markdown(
                    f"**{f} → {t}**  \n"
                    f"- Trigger: {row.get('trigger', '')}  \n"
                    f"- Response: *{row.get('action', '')}*"
                )

    uw = results.get("unusual_whales_context")
    if isinstance(uw, dict) and uw:
        st.subheader("Unusual Whales — Greeks & flow")
        g = uw.get("greeks_from_chain") or {}
        if g:
            st.markdown("**Chain Greeks (ATM selection)**")
            st.json({k: v for k, v in g.items() if v is not None})
        fs = uw.get("flow_summary")
        if fs:
            st.markdown("**Options flow (recent alerts)**")
            st.json(fs)
        cr = uw.get("selected_chain_columns")
        if cr:
            with st.expander("Selected chain snapshot"):
                st.json(cr)
        st.caption(f"Spot at first bar used for ATM: {uw.get('spot_at_entry_bar')}")

    trades = results.get("trades", [])
    transactions = results.get("transactions", [])
    if trades:
        st.subheader("Trades")
        st.dataframe(pd.DataFrame(trades), use_container_width=True)
    if transactions:
        st.subheader("Transaction Log")
        st.dataframe(pd.DataFrame(transactions), use_container_width=True)

    contributions = results.get("symbol_contributions", {})
    if contributions:
        st.subheader("Per-Symbol Contribution")
        st.dataframe(pd.DataFrame([{"symbol": k, "contribution": v} for k, v in contributions.items()]), use_container_width=True)

    regime_preview = results.get("regime_series")
    ohlcv_preview = results.get("ohlcv")
    if isinstance(regime_preview, pd.Series) and isinstance(ohlcv_preview, pd.DataFrame):
        render_regime_chart(ohlcv_preview, regime_preview)

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
    except PydanticValidationError:
        return
    st.caption(
        f"{cfg.symbols[0]} | {cfg.start_date} → {cfg.end_date} | {cfg.timeframe} | {cfg.trading_mode}"
    )

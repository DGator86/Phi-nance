"""Action handlers for Streamlit workbench user events."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError as PydanticValidationError

from app_streamlit.cache import load_historical_data
from app_streamlit.state import (
    AppState,
    set_config,
    set_error,
    set_form_errors,
    set_results,
    transition_to,
)
from phi.backtest import run_direct_backtest, run_portfolio_backtest
from phi.exceptions import BacktestError, DataFetchError, ValidationError
from phi.logging import get_logger
from phi.options import run_options_backtest
from phi.options.regime_playbook import (
    OptionsRegimePlaybook,
    load_options_regime_playbook,
    playbook_to_summary_dict,
)
from phi.options.unusual_whales_context import enrich_run_config_options_from_uw
from phi.regime import list_saved_detectors, load_detector
from phi.regime.train import train_regime_detector
from phi.run_config import RunConfig, RunHistory
from phi.utils.validation import (
    sanitize_run_id,
    sanitize_ticker,
    validate_date_bounds,
    validate_positive_number,
)

logger = get_logger(__name__)

REGIME_METHOD_MAP = {
    "HMM": "hmm",
    "KMeans": "kmeans",
    "Clustering (KMeans)": "kmeans",
    "GMM": "gmm",
}


def build_regime_boosts_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Map raw detector state keys to friendly regime labels for regime_weighted blending."""
    label_map: dict[str, str] = {str(k): str(v) for k, v in (payload.get("regime_label_map") or {}).items()}
    matrix = payload.get("regime_boost_matrix") or {}
    out: dict[str, dict[str, float]] = {}
    for raw_state, weights in matrix.items():
        if not isinstance(weights, dict):
            continue
        friendly = label_map.get(str(raw_state), str(raw_state))
        out[friendly] = {str(k): float(v) for k, v in weights.items()}
    return out


def load_detector_from_payload(payload: dict[str, Any]) -> Any:
    """Load persisted regime detector from sidebar selection."""
    path = payload.get("regime_selected_model_path")
    if not path or not str(path).strip():
        raise BacktestError("Select a saved regime model in the sidebar, or train one first.")
    return load_detector(str(path))


def _json_safe_results(results: dict[str, Any]) -> dict[str, Any]:
    """Convert Series/DataFrame so RunHistory JSON save does not rely on str()."""
    out: dict[str, Any] = {}
    for k, v in results.items():
        if isinstance(v, pd.Series):
            out[k] = {str(ix): str(val) for ix, val in v.items()}
        elif isinstance(v, pd.DataFrame):
            out[k] = v.reset_index().to_dict(orient="list")
        else:
            out[k] = v
    return out


def _strip_heavy_results_for_storage(results: dict[str, Any]) -> dict[str, Any]:
    """Drop large frames from the persisted run artifact."""
    slim = {k: v for k, v in results.items() if k not in {"ohlcv"}}
    return _json_safe_results(slim)


def validate_config_payload(payload: dict[str, Any]) -> list[str]:
    """Return human-friendly config validation errors."""
    errors: list[str] = []
    symbols = payload.get("symbols") or [payload.get("symbol", "")]
    cleaned_symbols: list[str] = []
    for sym in symbols:
        try:
            cleaned_symbols.append(sanitize_ticker(sym))
        except ValidationError as exc:
            errors.append(str(exc))
    if cleaned_symbols:
        payload["symbols"] = cleaned_symbols
        payload["symbol"] = cleaned_symbols[0]

    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    try:
        if isinstance(start_date, date):
            validate_date_bounds(start_date, name="Start date")
        if isinstance(end_date, date):
            validate_date_bounds(end_date, name="End date")
        if isinstance(start_date, date) and isinstance(end_date, date) and start_date > end_date:
            raise ValidationError("Start date must be before end date.")
    except ValidationError as exc:
        errors.append(str(exc))

    try:
        validate_positive_number(payload.get("initial_capital", 0), name="Initial capital")
    except ValidationError as exc:
        errors.append(str(exc))

    if not payload.get("indicators"):
        errors.append("Enable at least one indicator.")

    if payload.get("trading_mode") == "options":
        vkey = str(payload.get("vendor", "")).lower().replace("-", "_").replace(" ", "")
        if vkey not in ("unusual_whales", "unusualwhales"):
            errors.append(
                "Options mode requires data vendor 'unusual_whales' (ATM Greeks + options flow from Unusual Whales)."
            )
        for field_name in ("option_strike", "option_iv", "option_qty"):
            try:
                validate_positive_number(payload.get(field_name, 0), name=field_name)
            except ValidationError as exc:
                errors.append(str(exc))
        try:
            validate_positive_number(payload.get("option_rate", 0), name="option_rate", allow_zero=True)
        except ValidationError as exc:
            errors.append(str(exc))
    return errors


def build_run_config(payload: dict[str, Any]) -> RunConfig:
    """Convert UI payload into validated RunConfig."""
    enabled = {k: v for k, v in payload["indicators"].items() if v.get("enabled", False)}
    blend_weights = payload.get("blend_weights") or {}
    if not blend_weights and enabled:
        equal = round(1.0 / len(enabled), 4)
        blend_weights = dict.fromkeys(enabled, equal)
        drift = 1.0 - sum(blend_weights.values())
        if drift:
            first = next(iter(blend_weights))
            blend_weights[first] += drift

    option_params: dict[str, dict[str, Any]] = {}
    if payload["trading_mode"] == "options":
        sym = sanitize_ticker(payload["symbol"])
        option_params[sym] = {
            "option_type": payload["option_type"],
            "strike": payload["option_strike"],
            "expiry": payload["option_expiry"],
            "iv": payload["option_iv"],
            "r": payload["option_rate"],
            "quantity": int(payload["option_qty"]),
        }

    return RunConfig(
        symbols=payload.get("symbols") or [sanitize_ticker(payload["symbol"])],
        start_date=payload["start_date"],
        end_date=payload["end_date"],
        timeframe=payload["timeframe"],
        vendor=payload["vendor"],
        initial_capital=validate_positive_number(payload["initial_capital"], name="initial_capital"),
        trading_mode=payload["trading_mode"],
        indicators=enabled,
        blend_method=payload["blend_method"],
        blend_weights=blend_weights,
        option_params=option_params,
        regime_detector_params=payload.get("regime_detector_params"),
        regime_boosts=payload.get("regime_boosts"),
        allocation_strategy=payload.get("allocation_strategy", "equal_weight"),
        allocation_params=payload.get("allocation_params", {}),
        rebalance_frequency=payload.get("rebalance_frequency", "M"),
        rebalance_threshold=payload.get("rebalance_threshold"),
        options_regime_playbook=payload.get("options_regime_playbook"),
    )


def handle_train_regime_detector(
    payload: dict[str, Any],
    *,
    load_data_fn: Callable[..., pd.DataFrame] = load_historical_data,
) -> tuple[Any, pd.Series, str | None]:
    """Train selected regime detector and cache detector+predictions in session state."""
    method_label = str(payload.get("regime_method", "HMM"))
    method = REGIME_METHOD_MAP.get(method_label)
    if method is None:
        logger.warning("Unknown regime method %r, falling back to kmeans", method_label)
        method = "kmeans"

    data = load_data_fn(
        sanitize_ticker(payload["symbol"]),
        payload["start_date"].isoformat(),
        payload["end_date"].isoformat(),
        payload["timeframe"],
        payload["vendor"],
    )
    if data is None or data.empty:
        raise BacktestError("No data returned for selected configuration.")

    detector, path = train_regime_detector(
        data,
        method=method,
        n_regimes=int(payload.get("regime_n_states", 3)),
        window=int(payload.get("regime_window", 20)),
        save=True,
    )
    regime_series = detector.predict(data)
    st.session_state.regime_detector = detector
    st.session_state["regime_series"] = regime_series
    st.session_state.regime_model_path = str(path) if path else None
    return detector, regime_series, str(path) if path else None


def handle_run_backtest(
    payload: dict[str, Any],
    *,
    load_data_fn: Callable[..., pd.DataFrame] = load_historical_data,
    run_equity_fn: Callable[..., tuple[dict[str, Any], Any]] = run_direct_backtest,
    run_options_fn: Callable[..., dict[str, Any]] = run_options_backtest,
) -> dict[str, Any] | None:
    """Validate payload, execute the chosen backtest flow, and update UI state."""
    errors = validate_config_payload(payload)
    set_form_errors(errors)
    if errors:
        transition_to(AppState.CONFIGURING)
        return None

    uw_context: dict[str, Any] = {}
    try:
        cfg = build_run_config(payload)
        transition_to(AppState.RUNNING)

        data_map: dict[str, pd.DataFrame] = {}
        for sym in cfg.symbols:
            data = load_data_fn(
                sym,
                cfg.start_date.isoformat(),
                cfg.end_date.isoformat(),
                cfg.timeframe,
                cfg.vendor,
            )
            if data is None or data.empty:
                raise BacktestError(f"No data returned for symbol {sym}.")
            data_map[sym] = data

        if cfg.trading_mode == "options":
            cfg, uw_context = enrich_run_config_options_from_uw(cfg, data_map[cfg.symbols[0]])

        set_config(cfg.model_dump())

        if cfg.trading_mode == "options":
            playbook = load_options_regime_playbook()
            if cfg.options_regime_playbook:
                playbook = OptionsRegimePlaybook.model_validate(cfg.options_regime_playbook)

            opt_regime_series: pd.Series | None = None
            if payload.get("regime_enabled"):
                detector = load_detector_from_payload(payload)
                use_precomputed = bool(payload.get("regime_use_precomputed"))
                precomputed = st.session_state.get("regime_series")
                primary_ohlcv = data_map[cfg.symbols[0]]
                if use_precomputed and isinstance(precomputed, pd.Series) and not precomputed.empty:
                    opt_regime_series = precomputed.reindex(primary_ohlcv.index).ffill()
                else:
                    opt_regime_series = detector.predict(primary_ohlcv)
                    st.session_state["regime_series"] = opt_regime_series

            results = run_options_fn(
                cfg, data_map[cfg.symbols[0]], regime_series=opt_regime_series
            )
            results = {
                **dict(results),
                "options_regime_playbook": playbook_to_summary_dict(playbook),
            }
            if uw_context:
                results["unusual_whales_context"] = uw_context
        else:
            primary = data_map[cfg.symbols[0]]
            regime_series = None
            regime_boosts = None
            regime_detector = None
            if payload.get("regime_enabled"):
                detector = load_detector_from_payload(payload)
                use_precomputed = bool(payload.get("regime_use_precomputed"))
                precomputed = st.session_state.get("regime_series")
                if use_precomputed and isinstance(precomputed, pd.Series) and not precomputed.empty:
                    regime_series = precomputed
                else:
                    regime_series = detector.predict(primary)
                    st.session_state["regime_series"] = regime_series
                regime_boosts = build_regime_boosts_from_payload(payload)
                if bool(payload.get("regime_detect_on_the_fly", True)):
                    regime_detector = detector

            if len(cfg.symbols) > 1:
                results = run_portfolio_backtest(
                    data_dict=data_map,
                    indicators=cfg.indicators,
                    blend_weights=cfg.blend_weights,
                    blend_method=cfg.blend_method,
                    initial_capital=cfg.initial_capital,
                    allocation_strategy=cfg.allocation_strategy,
                    allocation_params=cfg.allocation_params or {},
                    rebalance_frequency=cfg.rebalance_frequency,
                    rebalance_threshold=cfg.rebalance_threshold,
                    regime_series=regime_series,
                )
                if regime_series is not None:
                    results["regime_series"] = regime_series
                    results["ohlcv"] = primary
            else:
                results, _ = run_equity_fn(
                    ohlcv=primary,
                    symbol=cfg.symbols[0],
                    indicators=cfg.indicators,
                    blend_weights=cfg.blend_weights,
                    blend_method=cfg.blend_method,
                    initial_capital=cfg.initial_capital,
                    regime_series=regime_series,
                    regime_label_map=payload.get("regime_label_map") if payload.get("regime_enabled") else None,
                    regime_boosts=regime_boosts,
                    regime_detector=regime_detector,
                )
                if regime_series is not None:
                    results["regime_series"] = regime_series
                    results["ohlcv"] = primary

        history = RunHistory()
        run_id = history.create_run(cfg)
        history.save_results(run_id, _strip_heavy_results_for_storage(dict(results)))
        results_with_run = {**dict(results), "run_id": run_id}
        set_results(results_with_run)
        return results_with_run
    except (PydanticValidationError, ValidationError) as exc:
        logger.warning("Run configuration validation failed: %s", exc)
        set_error("Invalid configuration. Please correct highlighted inputs.", debug=str(exc))
    except BacktestError as exc:
        logger.warning("Backtest validation failed: %s", exc)
        set_error(str(exc), debug=traceback.format_exc())
    except DataFetchError as exc:
        logger.warning("Data fetch failed: %s", exc)
        set_error(str(exc), debug=traceback.format_exc())
    except Exception:  # noqa: BLE001
        logger.exception("Backtest failed")
        set_error("Backtest failed. Please check inputs and try again.", debug=traceback.format_exc())
    return None


def handle_load_run(run_id: str) -> dict[str, Any] | None:
    """Load a historical run and publish it into session state."""
    try:
        safe_run_id = sanitize_run_id(run_id)
    except ValidationError as exc:
        set_error(str(exc))
        return None

    history = RunHistory()
    cfg = history.load_config(safe_run_id)
    results = history.load_results(safe_run_id)
    if cfg is None or results is None:
        set_error(f"Run '{safe_run_id}' could not be loaded.")
        return None

    payload = dict(results)
    payload["run_id"] = safe_run_id
    set_config(cfg.model_dump())
    set_results(payload)
    return payload

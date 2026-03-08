"""Action handlers for Streamlit workbench user events."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from datetime import date
from typing import Any

import pandas as pd
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
from phi.backtest import run_direct_backtest
from phi.exceptions import BacktestError, ValidationError
from phi.logging import get_logger
from phi.options import run_options_backtest
from phi.run_config import RunConfig, RunHistory
from phi.utils.validation import (
    sanitize_run_id,
    sanitize_ticker,
    validate_date_bounds,
    validate_positive_number,
)

logger = get_logger(__name__)


def validate_config_payload(payload: dict[str, Any]) -> list[str]:
    """Return human-friendly config validation errors."""
    errors: list[str] = []
    try:
        payload["symbol"] = sanitize_ticker(payload.get("symbol", ""))
    except ValidationError as exc:
        errors.append(str(exc))

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
        symbols=[sanitize_ticker(payload["symbol"])],
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
    )


def handle_run_backtest(
    payload: dict[str, Any],
    *,
    load_data_fn: Callable[..., pd.DataFrame] = load_historical_data,
    run_equity_fn: Callable[..., tuple[dict[str, Any], Any]] = run_direct_backtest,
    run_options_fn: Callable[..., dict[str, Any]] = run_options_backtest,
) -> dict[str, Any] | None:
    """Validate inputs, run selected backtest mode, and update state machine."""
    errors = validate_config_payload(payload)
    set_form_errors(errors)
    if errors:
        transition_to(AppState.CONFIGURING)
        return None

    try:
        cfg = build_run_config(payload)
        set_config(cfg.model_dump())
        transition_to(AppState.RUNNING)

        data = load_data_fn(
            cfg.symbols[0],
            cfg.start_date.isoformat(),
            cfg.end_date.isoformat(),
            cfg.timeframe,
            cfg.vendor,
        )

        if data is None or data.empty:
            raise BacktestError("No data returned for selected configuration.")

        if cfg.trading_mode == "options":
            results = run_options_fn(cfg, data)
        else:
            results, _ = run_equity_fn(
                ohlcv=data,
                symbol=cfg.symbols[0],
                indicators=cfg.indicators,
                blend_weights=cfg.blend_weights,
                blend_method=cfg.blend_method,
                initial_capital=cfg.initial_capital,
            )

        history = RunHistory()
        run_id = history.create_run(cfg)
        history.save_results(run_id, dict(results))
        results_with_run = {**dict(results), "run_id": run_id}
        set_results(results_with_run)
        return results_with_run
    except (PydanticValidationError, ValidationError) as exc:
        logger.warning("Run configuration validation failed: %s", exc)
        set_error("Invalid configuration. Please correct highlighted inputs.", debug=str(exc))
    except BacktestError as exc:
        logger.warning("Backtest validation failed: %s", exc)
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

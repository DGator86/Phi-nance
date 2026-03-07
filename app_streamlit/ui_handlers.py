"""Action handlers for Streamlit workbench user events."""

from __future__ import annotations

import traceback
from datetime import date
from typing import Any, Callable, Optional

import pandas as pd
from pydantic import ValidationError

from app_streamlit.cache import load_historical_data
from app_streamlit.state import AppState, set_config, set_error, set_form_errors, set_results, transition_to
from phi.backtest import run_direct_backtest
from phi.logging import get_logger
from phi.options import run_options_backtest
from phi.run_config import RunConfig, RunHistory

logger = get_logger(__name__)


def validate_config_payload(payload: dict[str, Any]) -> list[str]:
    """Return human-friendly config validation errors."""
    errors: list[str] = []
    if not payload.get("symbol"):
        errors.append("Symbol is required.")
    start_date = payload.get("start_date")
    end_date = payload.get("end_date")
    if isinstance(start_date, date) and isinstance(end_date, date) and start_date > end_date:
        errors.append("Start date must be before end date.")
    if float(payload.get("initial_capital", 0)) <= 0:
        errors.append("Initial capital must be greater than zero.")
    if not payload.get("indicators"):
        errors.append("Enable at least one indicator.")
    return errors


def build_run_config(payload: dict[str, Any]) -> RunConfig:
    """Convert UI payload into validated RunConfig."""
    enabled = {k: v for k, v in payload["indicators"].items() if v.get("enabled", False)}
    blend_weights = payload.get("blend_weights") or {}
    if not blend_weights and enabled:
        equal = round(1.0 / len(enabled), 4)
        blend_weights = {name: equal for name in enabled}
        drift = 1.0 - sum(blend_weights.values())
        if drift:
            first = next(iter(blend_weights))
            blend_weights[first] += drift

    option_params: dict[str, dict[str, Any]] = {}
    if payload["trading_mode"] == "options":
        sym = payload["symbol"]
        option_params[sym] = {
            "option_type": payload["option_type"],
            "strike": payload["option_strike"],
            "expiry": payload["option_expiry"],
            "iv": payload["option_iv"],
            "r": payload["option_rate"],
            "quantity": int(payload["option_qty"]),
        }

    return RunConfig(
        symbols=[payload["symbol"]],
        start_date=payload["start_date"],
        end_date=payload["end_date"],
        timeframe=payload["timeframe"],
        vendor=payload["vendor"],
        initial_capital=float(payload["initial_capital"]),
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
) -> Optional[dict[str, Any]]:
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
            raise ValueError("No data returned for selected configuration.")

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
    except ValidationError as exc:
        logger.warning("Run configuration validation failed: %s", exc)
        set_error("Invalid configuration. Please correct highlighted inputs.", debug=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backtest failed")
        set_error("Backtest failed. Please check inputs and try again.", debug=traceback.format_exc())
    return None


def handle_load_run(run_id: str) -> Optional[dict[str, Any]]:
    """Load a historical run and publish it into session state."""
    history = RunHistory()
    cfg = history.load_config(run_id)
    results = history.load_results(run_id)
    if cfg is None or results is None:
        set_error(f"Run '{run_id}' could not be loaded.")
        return None

    payload = dict(results)
    payload["run_id"] = run_id
    set_config(cfg.model_dump())
    set_results(payload)
    return payload

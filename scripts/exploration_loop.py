#!/usr/bin/env python3
"""Automated exploration loop for options strategies.

Iterates across a symbol universe, strategy classes, and parameter spaces,
executes options backtests, and persists each run through the existing
`run_options_backtest(..., record_results=True)` path.
"""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from phi.backtest.engine import run_options_backtest
from phi.phiai.auto_tune import tune_parameter_space

DEFAULT_UNIVERSE_PATH = "configs/universe.yaml"
DEFAULT_INITIAL_CASH = 100_000.0


def load_universe(config_path: str = DEFAULT_UNIVERSE_PATH) -> list[str]:
    """Load equities + indexes from a YAML universe file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {config_path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    equities = config.get("equities", []) or []
    indexes = config.get("indexes", []) or []
    universe = [str(symbol).upper() for symbol in [*equities, *indexes]]

    if not universe:
        raise ValueError(f"Universe file is empty: {config_path}")
    return universe


def get_strategy_class(strategy_name: str):
    """Resolve a strategy class dynamically from strategies.options_regime."""
    module = importlib.import_module("strategies.options_regime")
    try:
        return getattr(module, strategy_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown strategy class: {strategy_name}") from exc


# Search spaces aligned with constructor params in strategies/options_regime.py.
STRATEGY_SEARCH_SPACES: dict[str, dict[str, tuple[Any, Any] | list[Any]]] = {
    "LongCallStrategy": {
        "min_volume": (1, 100),
    },
    "BullPutSpreadStrategy": {
        "min_volume": (1, 100),
        "width_pct": (0.01, 0.20),
    },
    "BearCallSpreadStrategy": {
        "min_volume": (1, 100),
        "width_pct": (0.01, 0.20),
    },
    "LongStraddleStrategy": {
        "min_volume": (1, 100),
    },
    "IronCondorStrategy": {
        "min_volume": (1, 100),
        "wing_pct": (0.01, 0.20),
    },
}


def make_objective(
    strategy_class,
    symbol: str,
    train_start: str,
    train_end: str,
    initial_cash: float,
):
    """Build objective callable for parameter optimization.

    Objective = negative Sharpe ratio over training window.
    """

    def objective(params: dict[str, Any]) -> float:
        strategy = strategy_class(symbol=symbol, **params)
        result = run_options_backtest(
            strategy=strategy,
            symbols=[symbol],
            start_date=train_start,
            end_date=train_end,
            initial_cash=initial_cash,
            record_results=True,
        )

        sharpe = float(result.get("metrics", {}).get("sharpe_ratio", -10.0))
        if np.isnan(sharpe) or sharpe < -10.0:
            sharpe = -10.0
        return -sharpe

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated options strategy exploration")
    parser.add_argument("--universe", type=str, default=DEFAULT_UNIVERSE_PATH, help="Path to universe YAML file")
    parser.add_argument("--train-months", type=int, default=6, help="Months of training data")
    parser.add_argument("--val-months", type=int, default=1, help="Months of validation data")
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--n-calls", type=int, default=50, help="Optimization calls per strategy")
    parser.add_argument("--method", type=str, default="bayesian", choices=["bayesian", "genetic", "random"], help="PhiAI optimizer method")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    universe = load_universe(args.universe)
    print(f"Universe: {universe}")

    today = datetime.today()
    end_date = today.strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=args.val_months * 30)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=(args.train_months + args.val_months) * 30)).strftime("%Y-%m-%d")
    val_start = train_end
    val_end = end_date

    print(f"Training period: {train_start} to {train_end}")
    print(f"Validation period: {val_start} to {val_end}")

    for symbol in universe:
        print(f"\n=== Processing {symbol} ===")

        for strategy_name, space in STRATEGY_SEARCH_SPACES.items():
            print(f"  Optimizing {strategy_name}...")
            strategy_class = get_strategy_class(strategy_name)
            objective = make_objective(
                strategy_class=strategy_class,
                symbol=symbol,
                train_start=train_start,
                train_end=train_end,
                initial_cash=args.initial_cash,
            )

            best_params, best_score = tune_parameter_space(
                objective_func=objective,
                param_space=space,
                method=args.method,
                n_trials=args.n_calls,
                direction="minimize",
                seed=args.random_state,
            )
            print(f"    Best params: {best_params} (Sharpe={-best_score:.4f})")

            best_strategy = strategy_class(symbol=symbol, **best_params)
            val_result = run_options_backtest(
                strategy=best_strategy,
                symbols=[symbol],
                start_date=val_start,
                end_date=val_end,
                initial_cash=args.initial_cash,
                record_results=True,
            )
            val_sharpe = float(val_result.get("metrics", {}).get("sharpe_ratio", -10.0))
            print(f"    Validation Sharpe: {val_sharpe:.4f}")


if __name__ == "__main__":
    main()

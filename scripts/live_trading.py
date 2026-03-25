#!/usr/bin/env python3
"""Regime-aware live signal generation for options strategies."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from phi.data.fetchers import fetch
from phi.options.data_adapter import fetch_options_data
from phi.regime import get_detailed_regime_for_symbol
from lumibot_strategies.options_regime import STRATEGY_CLASS_MAP


def load_best_models(config_path: str) -> dict[str, dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Best-model map not found: {config_path}. "
            "Generate it with scripts/analyze_results_db.py first."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_underlying_price(symbol: str, as_of: str) -> float | None:
    end = pd.Timestamp(as_of).date()
    frame = fetch(symbol=symbol, start=end, end=end + pd.Timedelta(days=1), timeframe="1D", vendor="yfinance")
    if frame is None or frame.empty:
        return None
    return float(frame["close"].iloc[-1])


def get_trading_signals(symbol: str, as_of: str, best_models: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    regime = get_detailed_regime_for_symbol(symbol, as_of=as_of)
    model_info = best_models.get(regime)
    if model_info is None:
        print(f"No model configured for regime={regime}")
        return []

    strategy_name = model_info.get("strategy_name")
    strategy_cls = STRATEGY_CLASS_MAP.get(strategy_name)
    if strategy_cls is None and isinstance(strategy_name, str):
        strategy_cls = next((cls for cls in STRATEGY_CLASS_MAP.values() if cls.__name__ == strategy_name), None)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy in map: {strategy_name}")

    params = model_info.get("parameters", {}) or {}
    strategy = strategy_cls(symbol=symbol, **params)

    options_df = fetch_options_data(symbol, as_of, as_of)
    if options_df is None or options_df.empty:
        print(f"No options data for {symbol} on {as_of}")
        return []

    underlying_price = get_underlying_price(symbol, as_of)
    if underlying_price is None:
        print(f"No underlying data for {symbol} on {as_of}")
        return []

    return strategy.generate_signals(as_of, options_df, underlying_price)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-aware live options signals")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--date", default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--best-models", default="configs/best_models_per_regime.yaml")
    parser.add_argument("--execute", action="store_true", help="Placeholder flag for broker execution wiring")
    args = parser.parse_args()

    best_models = load_best_models(args.best_models)
    signals = get_trading_signals(args.symbol, args.date, best_models)

    print(f"Signals for {args.symbol} on {args.date}: {len(signals)}")
    for signal in signals:
        print(signal)
        if args.execute:
            print("  -> TODO: send order to broker adapter")


if __name__ == "__main__":
    main()

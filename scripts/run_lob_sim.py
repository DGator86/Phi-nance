#!/usr/bin/env python3
"""CLI runner for limit order book simulations."""

from __future__ import annotations

import argparse
import json
from itertools import islice

import pandas as pd

from phi.lob.data import iter_events, load_custom_csv, load_dukascopy_ticks, load_lobster_csv
from phi.lob.engine import LobSimEngine
from phi.lob.strategy import ImbalanceStrategy, MarketMakingStrategy
from phi.lob.synthetic import generate_synthetic_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LOB simulation")
    parser.add_argument("--data", default="synthetic", help="synthetic|lobster|dukascopy|custom")
    parser.add_argument("--path", default="", help="Path to input CSV for non-synthetic data")
    parser.add_argument("--strategy", default="market_making", help="market_making|imbalance")
    parser.add_argument("--params", default="{}", help="JSON parameters for source/strategy")
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--output", default="", help="Optional output JSON file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = json.loads(args.params)

    if args.data == "synthetic":
        events = islice(generate_synthetic_events(**{k: v for k, v in params.items() if k in {"start_price", "spread_bps", "arrival_rate", "volatility", "size_mean", "size_sigma", "seed"}}), args.max_events)
    else:
        if not args.path:
            raise ValueError("--path is required for non-synthetic data")
        if args.data == "lobster":
            df = load_lobster_csv(args.path)
        elif args.data == "dukascopy":
            df = load_dukascopy_ticks(args.path)
        elif args.data == "custom":
            column_map = params.get("column_map", {})
            df = load_custom_csv(args.path, column_map)
        else:
            raise ValueError(f"Unsupported data source: {args.data}")
        events = islice(iter_events(df), args.max_events)

    if args.strategy == "market_making":
        strategy = MarketMakingStrategy(
            spread_bps=float(params.get("mm_spread_bps", 5.0)),
            size=float(params.get("size", 1.0)),
        )
    elif args.strategy == "imbalance":
        strategy = ImbalanceStrategy(
            threshold=float(params.get("threshold", 0.25)),
            size=float(params.get("size", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    result = LobSimEngine(events, strategy).run()
    payload = {
        "metrics": result.metrics,
        "fills": [fill.__dict__ for fill in result.fills],
        "equity_points": len(result.equity_curve),
    }

    if args.output:
        pd.Series(payload).to_json(args.output)
    print(json.dumps(payload, default=str, indent=2))


if __name__ == "__main__":
    main()

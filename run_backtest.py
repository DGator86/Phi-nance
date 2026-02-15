#!/usr/bin/env python3
"""
Backtest Runner
---------------
Central entry point for running any strategy backtest.

Usage:
    python run_backtest.py                        # defaults: buy_and_hold
    python run_backtest.py --strategy momentum
    python run_backtest.py --strategy buy_and_hold --start 2022-01-01 --end 2024-01-01
    python run_backtest.py --strategy momentum --budget 50000 --benchmark QQQ
"""

import argparse
from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting

from strategies.bollinger import BollingerBands
from strategies.breakout import ChannelBreakout
from strategies.buy_and_hold import BuyAndHold
from strategies.dual_sma import DualSMACrossover
from strategies.macd import MACDStrategy
from strategies.mean_reversion import MeanReversion
from strategies.momentum import MomentumRotation
from strategies.rsi import RSIStrategy
from strategies.wyckoff import WyckoffStrategy

STRATEGIES = {
    "buy_and_hold": BuyAndHold,
    "momentum": MomentumRotation,
    "mean_reversion": MeanReversion,
    "rsi": RSIStrategy,
    "bollinger": BollingerBands,
    "macd": MACDStrategy,
    "dual_sma": DualSMACrossover,
    "breakout": ChannelBreakout,
    "wyckoff": WyckoffStrategy,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Lumibot backtest")
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES.keys(),
        default="buy_and_hold",
        help="Strategy to backtest (default: buy_and_hold)",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime(2020, 1, 1),
        help="Backtest start date YYYY-MM-DD (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=datetime(2024, 12, 31),
        help="Backtest end date YYYY-MM-DD (default: 2024-12-31)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100000,
        help="Starting cash budget (default: 100000)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Benchmark symbol for comparison (default: SPY)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_class = STRATEGIES[args.strategy]

    print(f"Running backtest: {args.strategy}")
    print(f"  Period: {args.start.date()} to {args.end.date()}")
    print(f"  Budget: ${args.budget:,.0f}")
    print(f"  Benchmark: {args.benchmark}")
    print()

    results = strategy_class.backtest(
        YahooDataBacktesting,
        args.start,
        args.end,
        budget=args.budget,
        benchmark_asset=args.benchmark,
    )

    return results


if __name__ == "__main__":
    main()

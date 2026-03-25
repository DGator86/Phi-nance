"""Walk-forward regime-to-strategy smoke test using the options backtest engine."""

from __future__ import annotations

import argparse
from datetime import timedelta

import pandas as pd

from phi.backtest.engine import run_options_backtest
from phi.regime import get_detailed_regime_for_symbol, strategies_for_regime
from lumibot_strategies.options_regime import STRATEGY_CLASS_MAP


class RegimeMappedStrategy:
    """Delegates signal generation to strategy classes selected by daily regime."""

    def __init__(self, symbol: str, lookback_days: int = 252) -> None:
        self.symbol = symbol
        self.lookback_days = lookback_days

    def generate_signals(self, current_date, options_df: pd.DataFrame, underlying_price: float):
        regime = get_detailed_regime_for_symbol(self.symbol, as_of=current_date, lookback_days=self.lookback_days)
        candidates = strategies_for_regime(regime)
        if not candidates:
            return []
        selected = candidates[0]
        cls = STRATEGY_CLASS_MAP.get(selected)
        if cls is None:
            return []
        return cls(symbol=self.symbol).generate_signals(current_date, options_df, underlying_price)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a regime-mapped options backtest.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-02-29")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = RegimeMappedStrategy(symbol=args.symbol)
    result = run_options_backtest(
        strategy=strategy,
        symbols=[args.symbol],
        start_date=args.start,
        end_date=args.end,
        initial_cash=args.initial_cash,
    )

    portfolio = result["portfolio"]
    closed = portfolio.trade_log
    pnl = sum(float(trade.get("pnl", 0.0)) for trade in closed)
    print(f"Closed trades: {len(closed)}")
    print(f"Realized PnL: {pnl:.2f}")
    print(f"Final cash: {portfolio.cash:.2f}")

    # show a tiny regime sample for sanity
    start_dt = pd.Timestamp(args.start).date()
    print("Sample regimes:")
    for offset in range(3):
        d = start_dt + timedelta(days=offset)
        print(f"  {d}: {get_detailed_regime_for_symbol(args.symbol, as_of=d)}")


if __name__ == "__main__":
    main()

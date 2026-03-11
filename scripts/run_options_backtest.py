"""Run a simple options backtest using the BasicOptionsStrategy."""

from __future__ import annotations

from phi.backtest.engine import run_options_backtest
from strategies.options_basic import BasicOptionsStrategy


def main() -> None:
    symbol = "SPY"
    start = "2023-01-01"
    end = "2023-01-31"
    initial_cash = 100_000

    strategy = BasicOptionsStrategy(symbol, threshold_delta=0.6)
    portfolio = run_options_backtest(strategy, [symbol], start, end, initial_cash)

    open_positions = [p for p in portfolio.option_positions if p.exit_date is None]
    print(f"Final cash: {portfolio.cash:.2f}")
    print(f"Open options: {len(open_positions)}")


if __name__ == "__main__":
    main()

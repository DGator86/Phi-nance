"""
Momentum Rotation Strategy
---------------------------
Rotates into the asset with the best recent momentum from a
configurable universe of symbols. Re-evaluates every `rebalance_days`.

Usage:
    python strategies/momentum.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy


class MomentumRotation(Strategy):
    parameters = {
        "symbols": ["SPY", "VEU", "AGG", "GLD"],
        "lookback_days": 20,
        "rebalance_days": 5,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.days_since_rebalance = 0
        self.current_asset = None

    def on_trading_iteration(self):
        symbols = self.parameters["symbols"]
        lookback = self.parameters["lookback_days"]
        rebalance_period = self.parameters["rebalance_days"]

        self.days_since_rebalance += 1

        if self.first_iteration or self.days_since_rebalance >= rebalance_period:
            self.days_since_rebalance = 0
            best = self._best_momentum_asset(symbols, lookback)

            if best and best != self.current_asset:
                if self.current_asset:
                    self.sell_all()

                price = self.get_last_price(best)
                quantity = int(self.portfolio_value // price)
                if quantity > 0:
                    order = self.create_order(best, quantity, "buy")
                    self.submit_order(order)
                    self.current_asset = best
                    self.log_message(
                        f"Rotated into {best} ({quantity} shares @ ${price:.2f})"
                    )

    def _best_momentum_asset(self, symbols, lookback):
        bars = self.get_bars(symbols, lookback + 1, timestep="day")
        best_symbol = None
        best_return = float("-inf")

        for asset, bar_data in bars.items():
            momentum = bar_data.get_momentum(num_periods=lookback)
            if momentum is not None and momentum > best_return:
                best_return = momentum
                best_symbol = asset.symbol

        return best_symbol

    def trace_stats(self, context, snapshot_before):
        return {
            "current_asset": self.current_asset or "None",
            "portfolio_value": self.portfolio_value,
        }


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    MomentumRotation.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )

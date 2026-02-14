"""
Buy-and-Hold Strategy
---------------------
Buys a single asset on the first trading iteration and holds it
for the entire backtest period. A simple baseline to compare other
strategies against.

Usage:
    python strategies/buy_and_hold.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy


class BuyAndHold(Strategy):
    parameters = {
        "symbol": "SPY",
    }

    def initialize(self):
        self.sleeptime = "1D"

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]

        if self.first_iteration:
            price = self.get_last_price(symbol)
            quantity = int(self.portfolio_value // price)
            if quantity > 0:
                order = self.create_order(symbol, quantity, "buy")
                self.submit_order(order)
                self.log_message(
                    f"Bought {quantity} shares of {symbol} at ${price:.2f}"
                )


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    BuyAndHold.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )

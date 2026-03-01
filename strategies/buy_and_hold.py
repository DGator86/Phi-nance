"""
Buy-and-Hold Strategy
---------------------
Buys a single asset on the first trading iteration and holds it
for the entire backtest period. A simple baseline to compare other
strategies against.

Prediction logic: Always predicts UP (permanent bullish bias).
This is the naive baseline — any useful strategy should beat it.

Usage:
    python strategies/buy_and_hold.py
"""

import os
from datetime import datetime

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
AlphaVantageBacktesting = AlphaVantageFixedDataSource
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class BuyAndHold(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        price = self.get_last_price(symbol)
        
        # Prediction: always bullish
        self.record_prediction(symbol, "UP", price)

        if self.first_iteration:
            if price is None:
                self.log_message(f"WARNING: Price is None for {symbol} at {self.get_datetime()}. Skipping.")
                return

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

    # Use Alpha Vantage as primary data source
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")

    BuyAndHold.run_backtest(
        datasource_class=AlphaVantageBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        api_key=av_api_key,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

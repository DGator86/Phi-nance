"""
Mean Reversion (SMA Crossover) Strategy
----------------------------------------
Buys when the asset's price drops below its simple moving average
(oversold) and sells when it rises above (overbought). A classic
mean-reversion / trend-following hybrid.

Prediction logic:
    Price < SMA  ->  predicts UP   (expect reversion upward)
    Price > SMA  ->  predicts DOWN (expect reversion downward)
    Price == SMA ->  NEUTRAL

Usage:
    python strategies/mean_reversion.py
"""

import os
from datetime import datetime

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource as AlphaVantageBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class MeanReversion(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "sma_period": 20,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        sma_period = self.parameters["sma_period"]

        bars = self.get_bars([symbol], sma_period + 1, timestep="day")
        if not bars or symbol not in [a.symbol for a in bars]:
            return

        asset_key = [a for a in bars if a.symbol == symbol][0]
        df = bars[asset_key].df
        if len(df) < sma_period:
            return

        sma = df["close"].rolling(window=sma_period).mean().iloc[-1]
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        # Record prediction based on SMA signal
        if current_price < sma:
            self.record_prediction(symbol, "UP", current_price)
        elif current_price > sma:
            self.record_prediction(symbol, "DOWN", current_price)
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)

        # Trading logic (unchanged)
        if current_price < sma and position is None:
            quantity = int(self.portfolio_value * 0.95 // current_price)
            if quantity > 0:
                order = self.create_order(symbol, quantity, "buy")
                self.submit_order(order)
                self.log_message(
                    f"BUY {symbol}: price ${current_price:.2f} < SMA ${sma:.2f}"
                )

        elif current_price > sma and position is not None:
            self.sell_all()
            self.log_message(
                f"SELL {symbol}: price ${current_price:.2f} > SMA ${sma:.2f}"
            )

    def trace_stats(self, context, snapshot_before):
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
        }


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    # Use Alpha Vantage as primary data source
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")

    MeanReversion.run_backtest(
        datasource_class=AlphaVantageBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        api_key=av_api_key,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

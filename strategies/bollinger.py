"""
Bollinger Bands Strategy
-------------------------
Uses Bollinger Bands (SMA +/- N standard deviations) to detect
overbought/oversold conditions.

Prediction logic:
    Price < lower band  ->  UP   (oversold, expect bounce)
    Price > upper band  ->  DOWN (overbought, expect pullback)
    Price in between    ->  NEUTRAL

Usage:
    python strategies/bollinger.py
"""

import os
from datetime import datetime

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from lumibot.backtesting import AlphaVantageDataBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class BollingerBands(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "bb_period": 20,
        "num_std": 2.0,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        period = self.parameters["bb_period"]
        num_std = self.parameters["num_std"]

        bars = self.get_bars([symbol], period + 1, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < period:
            return

        closes = df["close"]
        sma = closes.rolling(window=period).mean().iloc[-1]
        std = closes.rolling(window=period).std().iloc[-1]
        upper = sma + num_std * std
        lower = sma - num_std * std
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        if current_price < lower:
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif current_price > upper:
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    # Use Alpha Vantage as primary data source
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")

    BollingerBands.run_backtest(
        datasource_class=AlphaVantageDataBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        api_key=av_api_key,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

"""
RSI (Relative Strength Index) Strategy
---------------------------------------
Classic oscillator strategy. Predicts UP when RSI drops below the
oversold threshold (expecting a bounce) and DOWN when RSI exceeds the
overbought threshold (expecting a pullback).

Usage:
    python strategies/rsi.py
"""

import os
from datetime import datetime

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from lumibot.backtesting import AlphaVantageBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class RSIStrategy(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        period = self.parameters["rsi_period"]
        oversold = self.parameters["oversold"]
        overbought = self.parameters["overbought"]

        bars = self.get_bars([symbol], period + 2, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < period + 1:
            return

        rsi = self._compute_rsi(df["close"], period)
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        if rsi < oversold:
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif rsi > overbought:
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)

    @staticmethod
    def _compute_rsi(closes, period):
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean().iloc[-1]
        avg_loss = loss.rolling(window=period, min_periods=period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    # Use Alpha Vantage as primary data source
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")

    RSIStrategy.run_backtest(
        datasource_class=AlphaVantageBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        api_key=av_api_key,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

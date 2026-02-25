"""
MACD (Moving Average Convergence Divergence) Strategy
------------------------------------------------------
Uses the MACD line crossing above/below its signal line to predict
directional moves.

Prediction logic:
    MACD > Signal  ->  UP   (bullish crossover)
    MACD < Signal  ->  DOWN (bearish crossover)
    Equal          ->  NEUTRAL

Usage:
    python strategies/macd.py
"""

import os
from datetime import datetime

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource as AlphaVantageBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class MACDStrategy(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        fast = self.parameters["fast_period"]
        slow = self.parameters["slow_period"]
        sig_period = self.parameters["signal_period"]

        # Need enough bars for the slow EMA + signal smoothing
        needed = slow + sig_period + 5
        bars = self.get_bars([symbol], needed, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < needed - 5:
            return

        closes = df["close"]
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig_period, adjust=False).mean()

        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        if macd_val > signal_val:
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif macd_val < signal_val:
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

    MACDStrategy.run_backtest(
        datasource_class=AlphaVantageBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        api_key=av_api_key,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

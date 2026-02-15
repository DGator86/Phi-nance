"""
Dual Moving Average Crossover Strategy
----------------------------------------
Uses a fast SMA and a slow SMA. When the fast crosses above the slow
("golden cross") it predicts UP; when it crosses below ("death cross")
it predicts DOWN.

This is a trend-following approach — it bets that the current trend
will continue.

Usage:
    python strategies/dual_sma.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class DualSMACrossover(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "fast_period": 10,
        "slow_period": 50,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]

        bars = self.get_bars([symbol], slow_period + 2, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < slow_period:
            return

        closes = df["close"]
        sma_fast = closes.rolling(window=fast_period).mean().iloc[-1]
        sma_slow = closes.rolling(window=slow_period).mean().iloc[-1]
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        if sma_fast > sma_slow:
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif sma_fast < sma_slow:
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    DualSMACrossover.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )

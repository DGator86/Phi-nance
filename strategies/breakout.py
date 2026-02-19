"""
Price Channel Breakout Strategy
--------------------------------
Tracks the highest high and lowest low over a lookback window.
When price breaks above the upper channel, it predicts UP (momentum
breakout). When price breaks below the lower channel, predicts DOWN.

This is a classic Donchian / turtle-trading inspired approach.

Usage:
    python strategies/breakout.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class ChannelBreakout(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "channel_period": 20,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        period = self.parameters["channel_period"]

        bars = self.get_bars([symbol], period + 2, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < period + 1:
            return

        # Channel is based on all bars *except* the latest one
        historical = df.iloc[-(period + 1):-1]
        upper_channel = historical["high"].max()
        lower_channel = historical["low"].min()
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        if current_price > upper_channel:
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif current_price < lower_channel:
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    ChannelBreakout.run_backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

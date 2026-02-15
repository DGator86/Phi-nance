"""
Wyckoff Accumulation / Distribution Strategy
----------------------------------------------
Simplified Wyckoff analysis that identifies accumulation and
distribution phases using price position within its recent range
and volume patterns on up vs down days.

Core idea (Richard Wyckoff, 1930s):
    - **Accumulation**: price is in the lower part of its range AND
      volume is heavier on up-days than down-days (smart money buying).
      Predicts UP.
    - **Distribution**: price is in the upper part of its range AND
      volume is heavier on down-days than up-days (smart money selling).
      Predicts DOWN.
    - Otherwise: NEUTRAL (no clear phase).

Usage:
    python strategies/wyckoff.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class WyckoffStrategy(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "lookback": 30,
        "range_pct": 0.3,  # bottom/top 30% of range triggers signal
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        lookback = self.parameters["lookback"]
        range_pct = self.parameters["range_pct"]

        bars = self.get_bars([symbol], lookback + 1, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < lookback:
            return

        closes = df["close"]
        volumes = df["volume"]
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        # --- Price position within range ---
        range_high = closes.max()
        range_low = closes.min()
        price_range = range_high - range_low
        if price_range == 0:
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        position_in_range = (current_price - range_low) / price_range

        # --- Volume analysis: up-day volume vs down-day volume ---
        price_changes = closes.diff()
        up_mask = price_changes > 0
        down_mask = price_changes < 0

        up_volume = volumes[up_mask].mean() if up_mask.any() else 0
        down_volume = volumes[down_mask].mean() if down_mask.any() else 0

        # --- Wyckoff phase detection ---
        if position_in_range <= range_pct and up_volume > down_volume:
            # Accumulation: low in range + smart money buying on up days
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
                    self.log_message(
                        f"ACCUMULATION {symbol}: range pos {position_in_range:.0%}, "
                        f"up_vol/down_vol = {up_volume:.0f}/{down_volume:.0f}"
                    )

        elif position_in_range >= (1 - range_pct) and down_volume > up_volume:
            # Distribution: high in range + smart money selling on down days
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
                self.log_message(
                    f"DISTRIBUTION {symbol}: range pos {position_in_range:.0%}, "
                    f"up_vol/down_vol = {up_volume:.0f}/{down_volume:.0f}"
                )

        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)

    def trace_stats(self, context, snapshot_before):
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
        }


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    WyckoffStrategy.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )

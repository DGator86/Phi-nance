"""
Liquidity Pool Strategy
------------------------
Identifies clusters of resting liquidity above swing highs and below
swing lows, then predicts price will be drawn toward those pools.

Core idea (ICT / smart-money concept):
    Stop-losses and limit orders accumulate above recent swing highs
    (buy-side liquidity) and below recent swing lows (sell-side
    liquidity).  Price tends to sweep these pools before reversing.

Detection:
    1. Find swing highs/lows over a lookback window using a
       configurable number of bars on each side (swing_strength).
    2. Measure distance from current price to nearest pool above
       and below.
    3. If the closest pool is overhead and price is moving toward it
       (close > open), predict UP (price drawn to buy-side liquidity).
    4. If the closest pool is below and price is moving toward it
       (close < open), predict DOWN (price drawn to sell-side liquidity).
    5. After a sweep (price pierces a pool level), predict reversal:
       swept above a high -> DOWN, swept below a low -> UP.

Usage:
    python strategies/liquidity_pools.py
"""

import os
from datetime import datetime

# Suppress Lumibot credential checks by forcing backtesting mode
os.environ["IS_BACKTESTING"] = "True"

from lumibot.backtesting import AlphaVantageBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class LiquidityPoolStrategy(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        "lookback": 40,
        "swing_strength": 3,  # bars on each side to confirm a swing
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        lookback = self.parameters["lookback"]
        strength = self.parameters["swing_strength"]

        needed = lookback + strength + 1
        bars = self.get_bars([symbol], needed, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < needed - 1:
            return

        highs = df["high"]
        lows = df["low"]
        closes = df["close"]
        opens = df["open"]
        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        # --- Find swing highs and lows (liquidity pool levels) ---
        swing_highs = []
        swing_lows = []

        for i in range(strength, len(df) - strength):
            # Swing high: higher high than `strength` bars on each side
            if all(highs.iloc[i] >= highs.iloc[i - j] for j in range(1, strength + 1)) and \
               all(highs.iloc[i] >= highs.iloc[i + j] for j in range(1, strength + 1)):
                swing_highs.append(highs.iloc[i])

            # Swing low: lower low than `strength` bars on each side
            if all(lows.iloc[i] <= lows.iloc[i - j] for j in range(1, strength + 1)) and \
               all(lows.iloc[i] <= lows.iloc[i + j] for j in range(1, strength + 1)):
                swing_lows.append(lows.iloc[i])

        if not swing_highs and not swing_lows:
            self.record_prediction(symbol, "NEUTRAL", current_price)
            return

        # --- Find nearest pools above and below current price ---
        pools_above = sorted([h for h in swing_highs if h > current_price])
        pools_below = sorted([l for l in swing_lows if l < current_price], reverse=True)

        nearest_above = pools_above[0] if pools_above else None
        nearest_below = pools_below[0] if pools_below else None

        # --- Check for sweep (price already pierced a pool) ---
        today_high = highs.iloc[-1]
        today_low = lows.iloc[-1]
        today_close = closes.iloc[-1]
        today_open = opens.iloc[-1]

        # Swept buy-side liquidity (poked above a swing high) -> reversal DOWN
        swept_highs = [h for h in swing_highs if today_high >= h > today_close]
        # Swept sell-side liquidity (poked below a swing low) -> reversal UP
        swept_lows = [l for l in swing_lows if today_low <= l < today_close]

        if swept_highs:
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
                self.log_message(
                    f"SWEEP above {max(swept_highs):.2f} -> reversal DOWN"
                )
            return

        if swept_lows:
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
                    self.log_message(
                        f"SWEEP below {min(swept_lows):.2f} -> reversal UP"
                    )
            return

        # --- No sweep: predict price drawn toward nearest pool ---
        dist_above = (nearest_above - current_price) if nearest_above else float("inf")
        dist_below = (current_price - nearest_below) if nearest_below else float("inf")
        bullish_candle = today_close > today_open

        if dist_above <= dist_below and bullish_candle and nearest_above:
            # Closer pool is overhead and momentum is up -> drawn to buy-side
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif dist_below < dist_above and not bullish_candle and nearest_below:
            # Closer pool is below and momentum is down -> drawn to sell-side
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
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

    # Use Alpha Vantage as primary data source
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")

    LiquidityPoolStrategy.run_backtest(
        datasource_class=AlphaVantageBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        api_key=av_api_key,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

"""
Momentum Rotation Strategy
---------------------------
Rotates into the asset with the best recent momentum from a
configurable universe of symbols. Re-evaluates every `rebalance_days`.

Prediction logic: Predicts UP for the best-momentum asset,
DOWN for all others in the universe.

Usage:
    python strategies/momentum.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class MomentumRotation(PredictionMixin, Strategy):
    parameters = {
        "symbols": ["SPY", "VEU", "AGG", "GLD"],
        "lookback_days": 20,
        "rebalance_days": 5,
    }

    def initialize(self):
        self.sleeptime = "1D"
        self.days_since_rebalance = 0
        self.current_asset = None
        self._init_predictions()

    def on_trading_iteration(self):
        symbols = self.parameters["symbols"]
        lookback = self.parameters["lookback_days"]
        rebalance_period = self.parameters["rebalance_days"]

        self.days_since_rebalance += 1

        if self.first_iteration or self.days_since_rebalance >= rebalance_period:
            self.days_since_rebalance = 0
            momentums = self._get_momentums(symbols, lookback)

            if not momentums:
                return

            best = max(momentums, key=lambda m: m["return"])
            best_symbol = best["symbol"]

            # Record predictions: UP for best, DOWN for the rest
            for m in momentums:
                signal = "UP" if m["symbol"] == best_symbol else "DOWN"
                self.record_prediction(m["symbol"], signal, m["price"])

            if best_symbol != self.current_asset:
                if self.current_asset:
                    self.sell_all()

                price = best["price"]
                quantity = int(self.portfolio_value // price)
                if quantity > 0:
                    order = self.create_order(best_symbol, quantity, "buy")
                    self.submit_order(order)
                    self.current_asset = best_symbol
                    self.log_message(
                        f"Rotated into {best_symbol} ({quantity} shares @ ${price:.2f})"
                    )

    def _get_momentums(self, symbols, lookback):
        bars = self.get_bars(symbols, lookback + 1, timestep="day")
        results = []
        for asset, bar_data in bars.items():
            momentum = bar_data.get_momentum(num_periods=lookback)
            if momentum is not None:
                results.append({
                    "symbol": asset.symbol,
                    "price": bar_data.get_last_price(),
                    "return": momentum,
                })
        return results

    def trace_stats(self, context, snapshot_before):
        return {
            "current_asset": self.current_asset or "None",
            "portfolio_value": self.portfolio_value,
        }


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    MomentumRotation.run_backtest(
        datasource_class=YahooDataBacktesting,
        backtesting_start=backtesting_start,
        backtesting_end=backtesting_end,
        benchmark_asset="SPY",
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
    )

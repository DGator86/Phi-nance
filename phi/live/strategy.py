"""Live strategy base class and default implementation."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from phi.live.broker import BrokerOrder


class LiveStrategy:
    def __init__(self, config: dict[str, Any], broker: Any, portfolio: Any) -> None:
        self.config = config
        self.broker = broker
        self.portfolio = portfolio
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=64))

    def get_signals(self, bar: dict[str, Any]) -> dict[str, float]:
        symbol = str(bar["symbol"])
        close = float(bar["close"])
        hist = self.history[symbol]
        hist.append(close)
        if len(hist) < 2:
            return {symbol: 0.0}
        return {symbol: (hist[-1] / hist[-2]) - 1.0}

    def allocate(self, capital: float, signals: dict[str, float], prices: dict[str, float], regime: str | None = None) -> dict[str, float]:
        strategy = str(self.config.get("allocation_strategy", "equal_weight")).lower()
        if strategy in {"signal", "signal_weighted"}:
            total = sum(abs(v) for v in signals.values())
            return {s: abs(v) / total for s, v in signals.items()} if total > 0 else {s: 1.0 / len(prices) for s in prices}
        fixed = self.config.get("allocation_params", {}).get("weights", {})
        if strategy in {"fixed", "fixed_weight"} and fixed:
            return {s: float(fixed.get(s, 0.0)) for s in prices}
        return {s: 1.0 / len(prices) for s in prices}

    def on_bar(self, bar: dict[str, Any]) -> list[BrokerOrder]:
        symbol = str(bar["symbol"])
        price = float(bar["close"])
        signals = self.get_signals(bar)
        target_weights = self.allocate(self.portfolio.equity(), signals, {symbol: price}, regime=None)
        target_value = self.portfolio.equity() * float(target_weights.get(symbol, 0.0))
        current_qty = float(self.portfolio.positions.get(symbol).qty) if symbol in self.portfolio.positions else 0.0
        diff_value = target_value - (current_qty * price)
        qty = int(abs(diff_value) / price) if price > 0 else 0
        if qty <= 0:
            return []
        return [BrokerOrder(symbol=symbol, qty=qty, side="buy" if diff_value > 0 else "sell", order_type="market")]

"""Main live trading execution loop."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phi.config import settings
from phi.live.broker import AlpacaBroker, Broker, BrokerOrder
from phi.live.loader import load_live_config
from phi.live.portfolio import LivePortfolio
from phi.live.risk import RiskLimits, RiskManager
from phi.live.strategy import LiveStrategy
from phi.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LiveEngine:
    broker: Broker
    portfolio: LivePortfolio
    strategy: LiveStrategy
    risk_manager: RiskManager
    symbols: list[str]
    update_interval: int = 60
    running: bool = False
    bar_queue: "queue.Queue[dict[str, Any]]" = field(default_factory=queue.Queue)
    action_log: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls, config_path: Path | None = None) -> "LiveEngine":
        cfg = load_live_config(config_path=config_path, best_params_dir=settings.RUNS_DIR / "best_params")
        broker = AlpacaBroker(settings.BROKER_API_KEY, settings.BROKER_SECRET_KEY, settings.BROKER_BASE_URL)
        portfolio = LivePortfolio(initial_cash=float(cfg.get("initial_capital", 100000.0)))
        strategy = LiveStrategy(config=cfg, broker=broker, portfolio=portfolio)
        risk = RiskManager(RiskLimits())
        return cls(broker=broker, portfolio=portfolio, strategy=strategy, risk_manager=risk, symbols=list(settings.LIVE_SYMBOLS), update_interval=settings.LIVE_UPDATE_INTERVAL)

    def _on_bar(self, bar: dict[str, Any]) -> None:
        self.bar_queue.put(bar)

    def process_bar(self, bar: dict[str, Any]) -> None:
        symbol = str(bar["symbol"])
        price = float(bar["close"])
        self.portfolio.update_price(symbol, price)
        self.risk_manager.update_equity(self.portfolio.equity())
        for order in self.strategy.on_bar(bar):
            if not self.risk_manager.validate_order(order, equity=self.portfolio.equity(), price=price):
                self.action_log.append(f"risk_reject:{order.symbol}:{order.qty}")
                continue
            placed = self.broker.place_order(order)
            self.portfolio.apply_fill(order.symbol, order.qty, price, order.side)
            self.action_log.append(f"order:{placed.symbol}:{placed.side}:{placed.qty}")
        self.portfolio.record()

    def run(self, max_cycles: int | None = None) -> None:
        self.broker.connect()
        self.running = True
        cycles = 0
        try:
            while self.running:
                for symbol in self.symbols:
                    self.broker.subscribe_bars(symbol, self._on_bar)
                while not self.bar_queue.empty():
                    self.process_bar(self.bar_queue.get())
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(self.update_interval)
        finally:
            self.running = False
            self.broker.disconnect()

    def stop(self) -> None:
        self.running = False

"""
phinance.live
=============

Live and paper trading sub-package.

Sub-modules
-----------
  broker_base   — Abstract BrokerAdapter interface
  alpaca        — Alpaca Markets REST adapter (paper + live)
  ibkr          — Interactive Brokers (TWS/Gateway) adapter
  paper_engine  — Pure in-process paper trading simulation loop
  trading_loop  — Orchestrator: signal → decision → order
  order_models  — Order, Fill, Position dataclasses

Supported brokers
-----------------
  Alpaca    — https://alpaca.markets (requires alpaca-py>=0.20)
  IBKR      — Interactive Brokers TWS/Gateway (requires ib_insync or ibapi)
  Paper     — Built-in paper engine (no external deps, always available)

Quick start (Alpaca paper)
--------------------------
    from phinance.live import LiveTradingLoop
    from phinance.live.alpaca import AlpacaBroker

    broker = AlpacaBroker(
        api_key="...", secret_key="...",
        base_url="https://paper-api.alpaca.markets",
    )
    loop = LiveTradingLoop(broker=broker, symbol="SPY", capital=50_000)
    loop.run_once(ohlcv_df, indicators_cfg)
"""

from phinance.live.broker_base import BrokerAdapter, OrderSide, OrderType, OrderStatus
from phinance.live.order_models import Order, Fill, Position
from phinance.live.paper_engine import PaperBroker
from phinance.live.trading_loop import LiveTradingLoop

# Alpaca is optional (requires alpaca-py)
try:
    from phinance.live.alpaca import AlpacaBroker
    __all_brokers__ = ["AlpacaBroker", "PaperBroker"]
except ImportError:  # pragma: no cover
    AlpacaBroker = None  # type: ignore[assignment,misc]
    __all_brokers__ = ["PaperBroker"]

# IBKR is optional (requires ib_insync or ibapi)
try:
    from phinance.live.ibkr import IBKRBroker
    __all_brokers__.append("IBKRBroker")
except ImportError:  # pragma: no cover
    IBKRBroker = None  # type: ignore[assignment,misc]

from phinance.live.scheduler import TradingScheduler, SchedulerConfig, ScheduledTick, run_paper_scheduler

__all__ = [
    "BrokerAdapter",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "Order",
    "Fill",
    "Position",
    "PaperBroker",
    "LiveTradingLoop",
    "TradingScheduler",
    "SchedulerConfig",
    "ScheduledTick",
    "run_paper_scheduler",
    *__all_brokers__,
]

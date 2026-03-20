"""Order execution helper with kill-switch and risk checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OrderDecision:
    allowed: bool
    reason: str


class OrderManager:
    def __init__(self, broker: Any, risk_monitor: Any | None = None) -> None:
        self.broker = broker
        self.risk_monitor = risk_monitor
        self.kill_switch = False

    def set_kill_switch(self, enabled: bool) -> None:
        self.kill_switch = enabled

    def can_submit(self, symbol: str, side: str, qty: float) -> OrderDecision:
        if self.kill_switch:
            return OrderDecision(False, "kill switch enabled")
        if self.risk_monitor is not None:
            check = self.risk_monitor.pre_trade_check(symbol=symbol, side=side, qty=qty)
            if check is False:
                return OrderDecision(False, "risk monitor rejected order")
        return OrderDecision(True, "ok")

    def submit_order(self, **order_kwargs: Any) -> Any:
        decision = self.can_submit(
            symbol=str(order_kwargs.get("symbol", "")),
            side=str(order_kwargs.get("side", "")),
            qty=float(order_kwargs.get("qty", 0)),
        )
        if not decision.allowed:
            raise RuntimeError(f"Order blocked: {decision.reason}")
        return self.broker.safe_call("submit_order", **order_kwargs)

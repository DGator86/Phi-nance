"""Live trading engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LiveEngineState:
    running: bool = False
    halted_reason: str | None = None


class LiveEngine:
    def __init__(self, data_manager: Any, order_manager: Any, config: dict[str, Any] | None = None) -> None:
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.state = LiveEngineState()
        self.config = config or {}
        self.use_hierarchical = bool(self.config.get("use_hierarchical", False))
        self.meta_orchestrator: Any | None = None

    def set_meta_orchestrator(self, orchestrator: Any) -> None:
        self.meta_orchestrator = orchestrator

    def start(self) -> None:
        self.state.running = True
        self.state.halted_reason = None

    def stop(self, reason: str | None = None) -> None:
        self.state.running = False
        self.state.halted_reason = reason

    def _build_meta_state(self, quote: Any) -> np.ndarray:
        if isinstance(quote, dict):
            price = float(quote.get("price", quote.get("close", 0.0)))
            bid = float(quote.get("bid", price))
            ask = float(quote.get("ask", price))
            spread = ask - bid
            volume = float(quote.get("volume", 0.0))
            return np.array([price, spread, volume, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.zeros(10, dtype=np.float32)

    def tick(self, symbol: str) -> dict[str, Any]:
        if not self.state.running:
            return {"status": "stopped", "reason": self.state.halted_reason}

        try:
            quote = self.data_manager.fetch("quotes", symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            self.stop(reason=f"quotes unavailable: {exc}")
            return {"status": "halted", "reason": self.state.halted_reason}

        response: dict[str, Any] = {"status": "ok", "symbol": symbol, "quote": quote}

        if self.use_hierarchical and self.meta_orchestrator is not None:
            decision = self.meta_orchestrator.tick(
                {
                    "meta_state": self._build_meta_state(quote),
                    "market_state": {"volatility": 0.0, "regime": "sideways", "regime_value": 0.0},
                    "portfolio_state": {"drawdown": 0.0, "sharpe": 0.0},
                    "order": {},
                    "info": {},
                },
                deterministic=True,
            )
            response["meta_option"] = decision.option_name
            response["meta_action"] = decision.action

        return response

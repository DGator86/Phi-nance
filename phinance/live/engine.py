"""Live trading engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LiveEngineState:
    running: bool = False
    halted_reason: str | None = None


class LiveEngine:
    def __init__(self, data_manager: Any, order_manager: Any) -> None:
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.state = LiveEngineState()

    def start(self) -> None:
        self.state.running = True
        self.state.halted_reason = None

    def stop(self, reason: str | None = None) -> None:
        self.state.running = False
        self.state.halted_reason = reason

    def tick(self, symbol: str) -> dict[str, Any]:
        if not self.state.running:
            return {"status": "stopped", "reason": self.state.halted_reason}

        try:
            quote = self.data_manager.fetch("quotes", symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            self.stop(reason=f"quotes unavailable: {exc}")
            return {"status": "halted", "reason": self.state.halted_reason}

        return {"status": "ok", "symbol": symbol, "quote": quote}

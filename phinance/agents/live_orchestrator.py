"""Live trading orchestrator agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phinance.live.engine import LiveEngine


@dataclass
class OrchestratorStatus:
    running: bool
    message: str


class LiveOrchestrator:
    def __init__(self, engine: LiveEngine, risk_monitor: Any | None = None) -> None:
        self.engine = engine
        self.risk_monitor = risk_monitor

    def start(self) -> OrchestratorStatus:
        self.engine.start()
        return OrchestratorStatus(running=True, message="live orchestrator started")

    def stop(self, reason: str = "manual stop") -> OrchestratorStatus:
        self.engine.stop(reason=reason)
        return OrchestratorStatus(running=False, message=reason)

    def tick(self, symbol: str) -> dict[str, Any]:
        if self.risk_monitor is not None and hasattr(self.risk_monitor, "should_halt"):
            if self.risk_monitor.should_halt():
                self.engine.stop(reason="risk monitor halt")
                return {"status": "halted", "reason": "risk monitor halt"}
        return self.engine.tick(symbol)

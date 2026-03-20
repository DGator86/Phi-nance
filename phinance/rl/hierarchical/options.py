"""Option abstractions for hierarchical reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Protocol


class OptionPolicy(Protocol):
    """Simple low-level policy interface expected by options."""

    def act(self, state: Any, deterministic: bool = True) -> Any:
        ...


@dataclass
class Option:
    """Temporally extended action executed by a low-level policy."""

    name: str
    policy: OptionPolicy
    initiation_condition: Callable[[Dict[str, Any]], bool]
    termination_condition: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    max_steps: int = 1
    can_interrupt: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_initiate(self, context: Dict[str, Any]) -> bool:
        return bool(self.initiation_condition(context))

    def should_terminate(self, context: Dict[str, Any], info: Dict[str, Any]) -> bool:
        if int(context.get("option_elapsed", 0)) >= int(self.max_steps):
            return True
        return bool(self.termination_condition(context, info))

    def act(self, state: Any, deterministic: bool = True) -> Any:
        return self.policy.act(state, deterministic=deterministic)


class DoNothingPolicy:
    """Policy placeholder for an idle option."""

    def act(self, state: Any, deterministic: bool = True) -> int:  # noqa: ARG002
        return 0


def execution_initiation(context: Dict[str, Any]) -> bool:
    order = context.get("order", {})
    return bool(order and order.get("status", "open") in {"open", "partial"} and float(order.get("remaining_shares", 0.0)) > 0.0)


def execution_termination(context: Dict[str, Any], info: Dict[str, Any]) -> bool:
    order = context.get("order", {})
    if info.get("order_cancelled", False):
        return True
    remaining = float(info.get("remaining_shares", order.get("remaining_shares", 0.0)))
    return remaining <= 0.0


def strategy_rd_initiation(context: Dict[str, Any]) -> bool:
    period = max(int(context.get("strategy_interval", 10)), 1)
    step = int(context.get("global_step", 0))
    return step % period == 0


def strategy_rd_termination(context: Dict[str, Any], info: Dict[str, Any]) -> bool:  # noqa: ARG001
    return bool(info.get("strategy_proposed", True))


def risk_monitor_initiation(context: Dict[str, Any]) -> bool:  # noqa: ARG001
    return True


def risk_monitor_termination(context: Dict[str, Any], info: Dict[str, Any]) -> bool:  # noqa: ARG001
    return False

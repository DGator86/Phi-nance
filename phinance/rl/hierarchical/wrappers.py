"""Wrappers converting existing low-level agents into option-compatible policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from phinance.rl.hierarchical.options import (
    DoNothingPolicy,
    Option,
    execution_initiation,
    execution_termination,
    risk_monitor_initiation,
    risk_monitor_termination,
    strategy_rd_initiation,
    strategy_rd_termination,
)


@dataclass
class ExecutionOptionPolicy:
    agent: Any

    def act(self, state: Dict[str, Any], deterministic: bool = True) -> np.ndarray:  # noqa: ARG002
        order = state.get("order", {})
        market_data = state.get("market_data")
        if not isinstance(market_data, pd.DataFrame) or market_data.empty:
            market_data = pd.DataFrame(
                {"open": [100.0], "high": [100.1], "low": [99.9], "close": [100.0], "volume": [10_000.0]}
            )
        decision = self.agent.execute_order(order, market_data)
        remaining = max(float(order.get("remaining_shares", order.get("qty", 1.0))), 1.0)
        return np.array([decision.shares_to_trade / remaining, decision.urgency], dtype=np.float32)


@dataclass
class StrategyOptionPolicy:
    agent: Any

    def act(self, state: Dict[str, Any], deterministic: bool = True) -> Dict[str, Any]:
        market_state = state.get("market_state", {})
        return self.agent.propose_strategy(market_state=market_state, deterministic=deterministic)


@dataclass
class RiskOptionPolicy:
    agent: Any

    def act(self, state: Dict[str, Any], deterministic: bool = True) -> Dict[str, float]:  # noqa: ARG002
        portfolio_state = state.get("portfolio_state", {})
        market_state = state.get("market_state", {})
        return self.agent.get_risk_limits(portfolio_state=portfolio_state, market_data=market_state)


def build_default_options(config: Dict[str, Any] | None = None) -> list[Option]:
    from phinance.agents.execution import ExecutionAgent
    from phinance.agents.risk_monitor import RiskMonitorAgent
    from phinance.agents.strategy_rd import StrategyRDAgent

    cfg = config or {}
    execution_agent = ExecutionAgent(use_rl=bool(cfg.get("use_rl", True)), policy_path=str(cfg.get("execution_policy", "models/execution_agent/latest.pt")))
    strategy_agent = StrategyRDAgent(use_rl=bool(cfg.get("use_rl", True)), policy_path=str(cfg.get("strategy_policy", "models/strategy_rd_agent/latest.pt")))
    risk_agent = RiskMonitorAgent(use_rl=bool(cfg.get("use_rl", True)), policy_path=str(cfg.get("risk_policy", "models/risk_monitor_agent/latest.pt")))

    return [
        Option(
            name="execution",
            policy=ExecutionOptionPolicy(execution_agent),
            initiation_condition=execution_initiation,
            termination_condition=execution_termination,
            max_steps=int(cfg.get("execution_max_steps", 5)),
            can_interrupt=True,
        ),
        Option(
            name="strategy_rd",
            policy=StrategyOptionPolicy(strategy_agent),
            initiation_condition=strategy_rd_initiation,
            termination_condition=strategy_rd_termination,
            max_steps=1,
            can_interrupt=True,
        ),
        Option(
            name="risk_monitor",
            policy=RiskOptionPolicy(risk_agent),
            initiation_condition=risk_monitor_initiation,
            termination_condition=risk_monitor_termination,
            max_steps=int(cfg.get("risk_monitor_max_steps", 1000)),
            can_interrupt=False,
        ),
        Option(
            name="idle",
            policy=DoNothingPolicy(),
            initiation_condition=lambda _: True,
            termination_condition=lambda _context, _info: True,
            max_steps=1,
            can_interrupt=True,
        ),
    ]

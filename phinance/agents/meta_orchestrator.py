"""Runtime coordinator that uses a hierarchical meta-agent to switch low-level options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from phinance.rl.hierarchical.meta_agent import MetaAgent
from phinance.rl.hierarchical.options import Option
from phinance.rl.hierarchical.wrappers import build_default_options


@dataclass
class MetaDecision:
    option_name: str
    option_index: int
    action: Any


class MetaOrchestrator:
    """Lightweight runtime orchestrator for hierarchical policy inference."""

    def __init__(
        self,
        checkpoint: str | Path,
        options: list[Option] | None = None,
    ) -> None:
        self.meta_agent = MetaAgent.load(checkpoint)
        self.options = options or build_default_options({"use_rl": True})
        self.current_option_idx = len(self.options) - 1
        self.option_elapsed = 0
        self.global_step = 0

    def _meta_state(self, context: Dict[str, Any]) -> Any:
        meta_state = context.get("meta_state")
        if meta_state is None:
            raise ValueError("MetaOrchestrator requires 'meta_state' in context")
        return meta_state

    def _option_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        option_context = dict(context)
        option_context["option_elapsed"] = self.option_elapsed
        option_context["global_step"] = self.global_step
        return option_context

    def select_option(self, context: Dict[str, Any], deterministic: bool = True) -> int:
        meta_state = self._meta_state(context)
        desired_idx = self.meta_agent.act(meta_state, deterministic=deterministic)
        desired = self.options[desired_idx]
        active = self.options[self.current_option_idx]
        option_ctx = self._option_context(context)

        if desired_idx != self.current_option_idx and desired.can_initiate(option_ctx):
            if active.can_interrupt or active.should_terminate(option_ctx, context.get("info", {})):
                self.current_option_idx = desired_idx
                self.option_elapsed = 0
        return self.current_option_idx

    def tick(self, context: Dict[str, Any], deterministic: bool = True) -> MetaDecision:
        idx = self.select_option(context=context, deterministic=deterministic)
        option = self.options[idx]
        action = option.act(self._option_context(context), deterministic=deterministic)
        self.option_elapsed += 1
        self.global_step += 1

        if option.should_terminate(self._option_context(context), context.get("info", {})):
            self.option_elapsed = 0

        return MetaDecision(option_name=option.name, option_index=idx, action=action)

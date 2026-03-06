"""Runtime coordinator that uses a hierarchical meta-agent to switch low-level options."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from phinance.meta.vault_integration import load_discovered_templates
from phinance.rl.hierarchical.meta_agent import MetaAgent
from phinance.rl.hierarchical.options import Option
from phinance.rl.hierarchical.wrappers import build_default_options


class _DiscoveredStrategyPolicy:
    def __init__(self, template: Dict[str, Any]) -> None:
        self.template = template

    def act(self, state: Any, deterministic: bool = True) -> Dict[str, Any]:  # noqa: ARG002
        return self.template


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
        include_discovered_options: bool = False,
        strategy_vault_path: str = "data/strategy_vault.json",
    ) -> None:
        self.meta_agent = MetaAgent.load(checkpoint)
        self.options = options or build_default_options({"use_rl": True})
        if include_discovered_options:
            self.options.extend(self._build_discovered_options(strategy_vault_path))
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

    def _build_discovered_options(self, strategy_vault_path: str) -> list[Option]:
        templates = load_discovered_templates(strategy_vault_path)
        discovered_options: list[Option] = []
        for template in templates:
            strategy_id = template.get("params", {}).get("strategy_id", "unknown")
            discovered_options.append(
                Option(
                    name=f"discovered_{strategy_id}",
                    policy=_DiscoveredStrategyPolicy(template),
                    initiation_condition=lambda _ctx: True,
                    termination_condition=lambda _ctx, _info: True,
                    max_steps=1,
                    can_interrupt=True,
                    metadata={"source": "meta_gp"},
                )
            )
        return discovered_options

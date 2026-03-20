"""Advisor agent that converts trading state into natural language reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from phinance.llm.client import DummyLLMClient, create_client
from phinance.llm.prompts import load_prompt

logger = logging.getLogger(__name__)


def _compress_messages(messages: list[dict[str, str]], model: str | None = None) -> list[dict[str, str]]:
    """Attempt Headroom compression, otherwise pass-through."""
    try:
        from headroom import compress

        compressed = compress(messages, model=model)
        if hasattr(compressed, "messages"):
            return compressed.messages
        if isinstance(compressed, list):
            return compressed
    except Exception as exc:  # noqa: BLE001
        logger.debug("Headroom compression unavailable: %s", exc)
    return messages


class AdvisorAgent:
    def __init__(self, config_path: str = "configs/llm_config.yaml") -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.enabled = bool(self.config.get("enabled", True))
        self.client = create_client(self.config) if self.enabled else DummyLLMClient()

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            return {
                "enabled": False,
                "backend": "none",
                "model": "llama3",
                "temperature": 0.7,
                "max_tokens": 800,
            }
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data.get("llm", data)

    def _run_prompt(self, prompt_name: str, **kwargs: Any) -> str:
        prompt = load_prompt(prompt_name, **kwargs)
        messages = _compress_messages([{"role": "user", "content": prompt}], model=self.config.get("model"))

        try:
            return self.client.complete(
                messages,
                temperature=float(self.config.get("temperature", 0.7)),
                max_tokens=int(self.config.get("max_tokens", 800)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AdvisorAgent fallback due to LLM failure: %s", exc)
            return f"Advisor unavailable; fallback summary returned. Error: {exc}"

    def explain_trades(self, trades: list[dict[str, Any]], market_context: dict[str, Any]) -> str:
        return self._run_prompt(
            "llm-trade-explanation",
            trades=trades,
            market_context=market_context,
        )

    def risk_report(
        self,
        account_summary: dict[str, Any],
        positions: list[dict[str, Any]],
        risk_metrics: dict[str, Any],
    ) -> str:
        return self._run_prompt(
            "llm-risk-report",
            account_summary=account_summary,
            positions=positions,
            risk_metrics=risk_metrics,
        )

    def review_strategy(self, strategy_description: str, backtest_results: dict[str, Any]) -> str:
        return self._run_prompt(
            "llm-strategy-review",
            strategy_description=strategy_description,
            backtest_results=backtest_results,
        )

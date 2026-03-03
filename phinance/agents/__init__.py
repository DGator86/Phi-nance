"""
phinance.agents — Agentic AI layer for Phi-nance.

Sub-modules
-----------
  base          — Abstract AgentBase, AgentResult, AgentCapability
  rule_agent    — Deterministic rule-based agent (no LLM required)
  ollama_agent  — Free local LLM agent via Ollama server
  orchestrator  — AgentOrchestrator: multi-agent pipeline with backtest oversight

Quick start (rule-based, no dependencies)
-----------------------------------------
    from phinance.agents import run_with_agents
    result = run_with_agents(ohlcv_df)
    print(result.consensus_action, result.summary)

LLM agent (requires Ollama)
----------------------------
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ``ollama pull llama3.2``
    3. Start server:   ``ollama serve``

    from phinance.agents import OllamaAgent, check_ollama_ready
    if check_ollama_ready():
        agent = OllamaAgent()
        reply = agent.chat("Analyse this market: TREND_UP, signal +0.7")
"""

from phinance.agents.base import AgentBase, AgentResult, AgentCapability
from phinance.agents.rule_agent import RuleBasedAgent
from phinance.agents.ollama_agent import OllamaAgent, check_ollama_ready, list_ollama_models
from phinance.agents.orchestrator import (
    AgentOrchestrator,
    OrchestratorResult,
    run_with_agents,
)

__all__ = [
    # base interfaces
    "AgentBase",
    "AgentResult",
    "AgentCapability",
    # agents
    "RuleBasedAgent",
    "OllamaAgent",
    "check_ollama_ready",
    "list_ollama_models",
    # orchestrator
    "AgentOrchestrator",
    "OrchestratorResult",
    "run_with_agents",
]

"""
phinance.agents — LLM agent integrations.

Currently supports Ollama (free local LLM server).

Setup
-----
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:   ``ollama pull llama3.2``  (or plutus, gemma2, etc.)
  3. Start server:   ``ollama serve``  (usually auto-starts)

Usage
-----
    from phinance.agents import OllamaAgent, check_ollama_ready

    if check_ollama_ready():
        agent = OllamaAgent()
        reply = agent.chat("Analyse this market regime: TREND_UP with high vol")
        print(reply)
"""

from phinance.agents.ollama_agent import OllamaAgent, check_ollama_ready, list_ollama_models

__all__ = ["OllamaAgent", "check_ollama_ready", "list_ollama_models"]

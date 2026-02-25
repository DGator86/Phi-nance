"""
Phi-nance AI Agents
===================

Ollama-powered free local models for PhiAI, regime analysis, and Q&A.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:   ollama pull llama3.2  (or plutus, gemma2, etc.)
  3. Run Ollama:     ollama serve  (usually auto-starts)

Usage:
  from phi.agents import OllamaAgent
  agent = OllamaAgent()
  reply = agent.chat("Explain this regime")
"""

from .ollama_agent import OllamaAgent, list_ollama_models, check_ollama_ready

__all__ = ["OllamaAgent", "list_ollama_models", "check_ollama_ready"]

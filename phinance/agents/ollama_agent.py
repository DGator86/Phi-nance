"""
Ollama Agent — Free local LLM for Phi-nance

Uses Ollama API (http://localhost:11434) or ollama.com cloud.
Works with any model: llama3.2, gemma2, plutus, mistral, etc.

Docs: https://docs.ollama.com/api/introduction
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


def check_ollama_ready(host: str = "http://localhost:11434", timeout: int = 5) -> bool:
    """Check if Ollama is running and reachable."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models(host: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """List available models. Returns [] if Ollama unreachable."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("models", [])
    except Exception:
        return []


class OllamaAgent:
    """
    Simple Ollama chat client for Phi-nance.

    Parameters
    ----------
    model : str — model name (e.g. llama3.2, gemma2, 0xroyce/plutus)
    host  : str — Ollama URL (default localhost:11434)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self._chat_url = f"{self.host}/api/chat"
        self._generate_url = f"{self.host}/api/generate"

    def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a chat message and return the reply.

        Parameters
        ----------
        prompt : user message
        system : optional system prompt
        stream : if True, yield chunks (not implemented here)

        Returns
        -------
        str — model response
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        try:
            r = requests.post(
                self._chat_url,
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            return f"[Ollama error: {e}]"

    def generate(self, prompt: str) -> str:
        """Simple generate (no chat history)."""
        try:
            r = requests.post(
                self._generate_url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "")
        except Exception as e:
            return f"[Ollama error: {e}]"

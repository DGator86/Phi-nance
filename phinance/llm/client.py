"""Backend-agnostic LLM clients used by AdvisorAgent."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import requests


class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Send a chat completion request and return assistant text."""


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return str(body.get("message", {}).get("content", "")).strip()


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI backend selected but openai package is not installed. "
                "Install with `pip install openai`."
            ) from exc
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response.choices[0].message.content or "").strip()


class DummyLLMClient(LLMClient):
    """Offline fallback so advisor never blocks trading."""

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        _ = (messages, temperature, max_tokens)
        return "LLM advisor is unavailable; generated fallback explanation."


def create_client(config: dict[str, Any]) -> LLMClient:
    backend = str(config.get("backend", "ollama")).lower()

    if backend == "none":
        return DummyLLMClient()

    if backend == "ollama":
        return OllamaClient(
            model=str(config.get("model", "llama3")),
            base_url=str(config.get("base_url", "http://localhost:11434")),
            timeout=int(config.get("timeout", 60)),
        )

    if backend == "openai":
        raw_api_key = config.get("api_key")
        api_key = raw_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI backend requires `api_key` or OPENAI_API_KEY env var.")
        return OpenAIClient(api_key=str(api_key), model=str(config.get("model", "gpt-4o-mini")))

    raise ValueError(f"Unknown LLM backend: {backend}")

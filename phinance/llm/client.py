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
        timeframe_minutes: int = 1,
    ) -> str:
        """Send a chat completion request and return assistant text.

        Parameters
        ----------
        messages            : OpenAI-style message list
        temperature         : sampling temperature
        max_tokens          : maximum tokens to generate
        timeframe_minutes   : bar size in minutes — used by TieredLLMClient
                              to route between fast (Ollama) and slow (Anthropic)
                              backends.  Ignored by single-backend clients.
        """


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeframe_minutes: int = 1,
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


class AnthropicClient(LLMClient):
    """Claude via the Anthropic API — used for high-timeframe / regime calls."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        self.api_key = api_key
        self.model = model
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic backend selected but anthropic package is not installed. "
                "Install with `pip install anthropic`."
            ) from exc
        self._client = _anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeframe_minutes: int = 1,
    ) -> str:
        import anthropic as _anthropic

        # Separate system message (Anthropic API takes it as a top-level param)
        system_content = ""
        user_messages: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                user_messages.append(m)

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=user_messages,
        )
        if system_content:
            kwargs["system"] = system_content

        try:
            response = self._client.messages.create(**kwargs)
            return str(response.content[0].text).strip()
        except _anthropic.APIError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc


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
        timeframe_minutes: int = 1,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(response.choices[0].message.content or "").strip()


class TieredLLMClient(LLMClient):
    """Routes requests to different backends based on bar timeframe.

    Routing rules (configurable via ``fast_threshold_minutes``):
      - timeframe <= fast_threshold_minutes  → ``fast_client``  (Ollama/DeepSeek)
      - timeframe >  fast_threshold_minutes  → ``slow_client``  (Anthropic/Claude)

    The regime qualifier always uses the slow (smarter) client regardless of
    timeframe because regime mis-classification is more costly than latency.

    Parameters
    ----------
    fast_client             : LLMClient for 1-5 min bars (free, local)
    slow_client             : LLMClient for 15 min+ bars and regime calls (smarter)
    fast_threshold_minutes  : bars at or below this value use fast_client (default 5)
    """

    def __init__(
        self,
        fast_client: LLMClient,
        slow_client: LLMClient,
        fast_threshold_minutes: int = 5,
    ) -> None:
        self.fast_client = fast_client
        self.slow_client = slow_client
        self.fast_threshold_minutes = fast_threshold_minutes

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeframe_minutes: int = 1,
    ) -> str:
        client = (
            self.fast_client
            if timeframe_minutes <= self.fast_threshold_minutes
            else self.slow_client
        )
        return client.complete(messages, temperature=temperature, max_tokens=max_tokens)

    def complete_regime(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        """Always use the slow (smarter) client for regime qualification."""
        return self.slow_client.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )


class DummyLLMClient(LLMClient):
    """Offline fallback so advisor never blocks trading."""

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        timeframe_minutes: int = 1,
    ) -> str:
        _ = (messages, temperature, max_tokens, timeframe_minutes)
        return "LLM advisor is unavailable; generated fallback explanation."


def create_client(config: dict[str, Any]) -> LLMClient:
    """Factory — builds the appropriate LLMClient from a config dict.

    Supported backends
    ------------------
    ``ollama``   : free local model via Ollama (default)
    ``anthropic``: Claude via Anthropic API  (requires ANTHROPIC_API_KEY)
    ``openai``   : OpenAI Chat Completions   (requires OPENAI_API_KEY)
    ``tiered``   : OllamaClient for fast TF + AnthropicClient for slow TF/regime
    ``none``     : DummyLLMClient (offline fallback)
    """
    backend = str(config.get("backend", "ollama")).lower()

    if backend == "none":
        return DummyLLMClient()

    if backend == "ollama":
        return OllamaClient(
            model=str(config.get("model", "deepseek-r1:7b")),
            base_url=str(config.get("base_url", "http://localhost:11434")),
            timeout=int(config.get("timeout", 120)),
        )

    if backend == "anthropic":
        raw_api_key = config.get("api_key")
        api_key = raw_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic backend requires `api_key` or ANTHROPIC_API_KEY env var."
            )
        return AnthropicClient(
            api_key=str(api_key),
            model=str(config.get("model", "claude-haiku-4-5-20251001")),
        )

    if backend == "openai":
        raw_api_key = config.get("api_key")
        api_key = raw_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI backend requires `api_key` or OPENAI_API_KEY env var.")
        return OpenAIClient(api_key=str(api_key), model=str(config.get("model", "gpt-4o-mini")))

    if backend == "tiered":
        # Fast client — Ollama / DeepSeek for 1-5 min bars
        fast_cfg = config.get("fast", {})
        fast_client = OllamaClient(
            model=str(fast_cfg.get("model", "deepseek-r1:7b")),
            base_url=str(fast_cfg.get("base_url", "http://localhost:11434")),
            timeout=int(fast_cfg.get("timeout", 120)),
        )

        # Slow client — Anthropic Claude for higher TF + regime
        slow_cfg = config.get("slow", {})
        raw_api_key = slow_cfg.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not raw_api_key:
            # Fall back to second Ollama client when no Anthropic key available
            slow_client: LLMClient = OllamaClient(
                model=str(slow_cfg.get("model", "deepseek-r1:7b")),
                base_url=str(slow_cfg.get("base_url", "http://localhost:11434")),
                timeout=int(slow_cfg.get("timeout", 180)),
            )
        else:
            slow_client = AnthropicClient(
                api_key=str(raw_api_key),
                model=str(slow_cfg.get("model", "claude-haiku-4-5-20251001")),
            )

        return TieredLLMClient(
            fast_client=fast_client,
            slow_client=slow_client,
            fast_threshold_minutes=int(config.get("fast_threshold_minutes", 5)),
        )

    raise ValueError(f"Unknown LLM backend: {backend}")

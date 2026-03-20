from __future__ import annotations

import pytest

from phinance.llm.client import DummyLLMClient, OllamaClient, create_client


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "hello from ollama"}}


def test_ollama_client_complete(monkeypatch):
    def _fake_post(url, json, timeout):
        assert url.endswith("/api/chat")
        assert json["model"] == "llama3"
        assert timeout == 10
        return _FakeResponse()

    monkeypatch.setattr("phinance.llm.client.requests.post", _fake_post)
    client = OllamaClient(model="llama3", timeout=10)
    text = client.complete([{"role": "user", "content": "hi"}])
    assert text == "hello from ollama"


def test_create_client_none_backend_returns_dummy():
    client = create_client({"backend": "none"})
    assert isinstance(client, DummyLLMClient)


def test_create_client_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        create_client({"backend": "openai", "model": "gpt-4o-mini", "api_key": ""})

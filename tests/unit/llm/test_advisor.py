from __future__ import annotations

from pathlib import Path

from phinance.llm.advisor import AdvisorAgent


class _StubClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages, temperature=0.7, max_tokens=500):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return "stubbed response"


def test_advisor_explain_trades_renders_prompt(tmp_path):
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        """
llm:
  enabled: true
  backend: none
  model: llama3
  temperature: 0.2
  max_tokens: 111
""".strip(),
        encoding="utf-8",
    )

    advisor = AdvisorAgent(config_path=str(cfg))
    stub = _StubClient()
    advisor.client = stub

    out = advisor.explain_trades(
        trades=[{"symbol": "SPY", "side": "buy", "qty": 1}],
        market_context={"regime": "trend_up"},
    )
    assert out == "stubbed response"
    assert stub.calls
    content = stub.calls[0]["messages"][0]["content"]
    assert "SPY" in content
    assert "trend_up" in content


def test_advisor_fallback_on_client_error(tmp_path):
    cfg = tmp_path / "llm.yaml"
    cfg.write_text("llm:\n  enabled: true\n  backend: none\n", encoding="utf-8")

    advisor = AdvisorAgent(config_path=str(cfg))

    class _ErrClient:
        def complete(self, messages, temperature=0.7, max_tokens=500):
            raise RuntimeError("boom")

    advisor.client = _ErrClient()
    out = advisor.review_strategy("x", {"sharpe": 1.2})
    assert "fallback" in out.lower()

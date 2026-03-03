"""
tests/unit/test_agents.py
==========================

Unit tests for phinance.agents:
  - AgentBase / AgentResult / AgentCapability (base)
  - RuleBasedAgent (rule_agent)
  - OllamaAgent (mocked — no Ollama server needed)
  - AgentOrchestrator (orchestrator)
  - run_with_agents convenience wrapper

All external services (Ollama HTTP) are mocked.
"""

from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.ohlcv import make_ohlcv

from phinance.agents.base import AgentBase, AgentCapability, AgentResult
from phinance.agents.rule_agent import RuleBasedAgent
from phinance.agents.ollama_agent import OllamaAgent, check_ollama_ready, list_ollama_models
from phinance.agents.orchestrator import (
    AgentOrchestrator,
    OrchestratorResult,
    run_with_agents,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _trending_ohlcv(n: int = 200) -> pd.DataFrame:
    """Steadily rising OHLCV (uses default make_ohlcv with cumsum walk)."""
    return make_ohlcv(n)


def _flat_ohlcv(n: int = 200) -> pd.DataFrame:
    """Flat OHLCV — constant price, no trend noise."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open":   100.0,
            "high":   101.0,
            "low":    99.0,
            "close":  100.0,
            "volume": 500_000.0,
        },
        index=idx,
    )


# Concrete subclass for abstract-base tests
class _ConcreteAgent(AgentBase):
    @property
    def name(self) -> str:
        return "ConcreteTestAgent"

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        return AgentResult(agent=self.name, action="hold", confidence=0.5, rationale="test")


# ─────────────────────────────────────────────────────────────────────────────
#  AgentResult
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentResult:

    def test_valid_buy(self):
        r = AgentResult(agent="X", action="buy", confidence=0.8, rationale="strong trend")
        assert r.action == "buy"
        assert r.confidence == 0.8

    def test_valid_sell(self):
        r = AgentResult(agent="X", action="sell", confidence=0.6, rationale="bearish")
        assert r.action == "sell"

    def test_valid_hold(self):
        r = AgentResult(agent="X", action="hold", confidence=0.4, rationale="neutral")
        assert r.action == "hold"

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action"):
            AgentResult(agent="X", action="fly", confidence=0.5, rationale="bad")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            AgentResult(agent="X", action="buy", confidence=1.1, rationale="over")

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            AgentResult(agent="X", action="buy", confidence=-0.1, rationale="neg")

    def test_to_dict_keys(self):
        r = AgentResult(agent="X", action="buy", confidence=0.7, rationale="test")
        d = r.to_dict()
        assert set(d.keys()) == {"agent", "action", "confidence", "rationale", "data", "timestamp"}

    def test_signal_value_buy(self):
        r = AgentResult(agent="X", action="buy", confidence=0.8, rationale="up")
        assert abs(r.signal_value - 0.8) < 1e-9

    def test_signal_value_sell(self):
        r = AgentResult(agent="X", action="sell", confidence=0.6, rationale="dn")
        assert abs(r.signal_value - (-0.6)) < 1e-9

    def test_signal_value_hold(self):
        r = AgentResult(agent="X", action="hold", confidence=0.9, rationale="flat")
        assert r.signal_value == 0.0

    def test_timestamp_set(self):
        before = time.time()
        r = AgentResult(agent="X", action="hold", confidence=0.5, rationale="t")
        after = time.time()
        assert before <= r.timestamp <= after

    def test_data_field_default_empty_dict(self):
        r = AgentResult(agent="X", action="hold", confidence=0.5, rationale="t")
        assert r.data == {}

    def test_data_field_custom(self):
        r = AgentResult(agent="X", action="buy", confidence=0.7, rationale="t",
                        data={"regime": "TREND_UP"})
        assert r.data["regime"] == "TREND_UP"


# ─────────────────────────────────────────────────────────────────────────────
#  AgentCapability
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentCapability:

    def test_all_capabilities_have_string_value(self):
        for cap in AgentCapability:
            assert isinstance(cap.value, str)

    def test_backtest_oversight_exists(self):
        assert AgentCapability.BACKTEST_OVERSIGHT

    def test_natural_language_exists(self):
        assert AgentCapability.NATURAL_LANGUAGE


# ─────────────────────────────────────────────────────────────────────────────
#  AgentBase (abstract)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentBase:

    def test_concrete_subclass_instantiates(self):
        agent = _ConcreteAgent()
        assert agent.name == "ConcreteTestAgent"

    def test_health_check_default_true(self):
        agent = _ConcreteAgent()
        assert agent.health_check() is True

    def test_capabilities_default_market_analysis(self):
        agent = _ConcreteAgent()
        assert AgentCapability.MARKET_ANALYSIS in agent.capabilities

    def test_repr_contains_name(self):
        agent = _ConcreteAgent()
        assert "ConcreteTestAgent" in repr(agent)

    def test_analyze_returns_agent_result(self):
        agent = _ConcreteAgent()
        result = agent.analyze({})
        assert isinstance(result, AgentResult)


# ─────────────────────────────────────────────────────────────────────────────
#  RuleBasedAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleBasedAgent:

    def _agent(self) -> RuleBasedAgent:
        return RuleBasedAgent()

    # identity / interface
    def test_name(self):
        assert self._agent().name == "RuleBasedAgent"

    def test_has_trade_signal_capability(self):
        assert AgentCapability.TRADE_SIGNAL in self._agent().capabilities

    def test_health_check_true(self):
        assert self._agent().health_check() is True

    # buy signals
    def test_strong_positive_signal_buys(self):
        r = self._agent().analyze({"signal": 0.8, "regime": "TREND_UP"})
        assert r.action == "buy"

    def test_above_threshold_neutral_regime_buys(self):
        r = self._agent().analyze({"signal": 0.5, "regime": "RANGE"})
        assert r.action == "buy"

    def test_buy_confidence_bounded(self):
        r = self._agent().analyze({"signal": 1.0, "regime": "TREND_UP"})
        assert 0.0 <= r.confidence <= 1.0

    def test_bullish_regime_boosts_buy_confidence(self):
        r_trend = self._agent().analyze({"signal": 0.5, "regime": "TREND_UP"})
        r_range = self._agent().analyze({"signal": 0.5, "regime": "RANGE"})
        assert r_trend.confidence >= r_range.confidence

    # sell signals
    def test_strong_negative_signal_sells(self):
        r = self._agent().analyze({"signal": -0.8, "regime": "TREND_DN"})
        assert r.action == "sell"

    def test_bearish_regime_boosts_sell_confidence(self):
        r_trend = self._agent().analyze({"signal": -0.5, "regime": "TREND_DN"})
        r_range = self._agent().analyze({"signal": -0.5, "regime": "RANGE"})
        assert r_trend.confidence >= r_range.confidence

    def test_sell_confidence_bounded(self):
        r = self._agent().analyze({"signal": -1.0, "regime": "TREND_DN"})
        assert 0.0 <= r.confidence <= 1.0

    # hold signals
    def test_near_zero_signal_holds(self):
        r = self._agent().analyze({"signal": 0.0, "regime": "RANGE"})
        assert r.action == "hold"

    def test_small_positive_holds(self):
        r = self._agent().analyze({"signal": 0.1, "regime": "RANGE"})
        assert r.action == "hold"

    def test_small_negative_holds(self):
        r = self._agent().analyze({"signal": -0.1, "regime": "RANGE"})
        assert r.action == "hold"

    def test_high_vol_regime_always_holds_for_neutral_signal(self):
        r = self._agent().analyze({"signal": 0.05, "regime": "HIGHVOL"})
        assert r.action == "hold"

    # risk gate
    def test_bad_sharpe_dampens_confidence(self):
        r_good = self._agent().analyze({"signal": 0.8, "regime": "TREND_UP",
                                         "backtest": {"sharpe": 1.5}})
        r_bad  = self._agent().analyze({"signal": 0.8, "regime": "TREND_UP",
                                         "backtest": {"sharpe": -2.0}})
        assert r_bad.confidence <= r_good.confidence

    # result properties
    def test_result_has_rationale(self):
        r = self._agent().analyze({"signal": 0.8, "regime": "TREND_UP"})
        assert len(r.rationale) > 0

    def test_result_data_contains_signal_and_regime(self):
        r = self._agent().analyze({"signal": 0.7, "regime": "BREAKOUT_UP"})
        assert "signal" in r.data
        assert "regime" in r.data

    def test_custom_thresholds(self):
        agent = RuleBasedAgent(buy_threshold=0.6, sell_threshold=-0.6)
        # Signal 0.5 — below custom buy threshold → should hold
        r = agent.analyze({"signal": 0.5, "regime": "RANGE"})
        assert r.action == "hold"


# ─────────────────────────────────────────────────────────────────────────────
#  OllamaAgent (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaAgent:

    @patch("phinance.agents.ollama_agent.requests.post")
    def test_chat_returns_string(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": "TREND_UP → buy"}},
        )
        mock_post.return_value.raise_for_status = lambda: None
        agent = OllamaAgent(model="llama3.2")
        reply = agent.chat("Analyse the market.")
        assert isinstance(reply, str)
        assert len(reply) > 0

    @patch("phinance.agents.ollama_agent.requests.post")
    def test_generate_returns_string(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "Some analysis"},
        )
        mock_post.return_value.raise_for_status = lambda: None
        agent = OllamaAgent()
        result = agent.generate("What is RSI?")
        assert isinstance(result, str)

    @patch("phinance.agents.ollama_agent.requests.post", side_effect=ConnectionError("refused"))
    def test_chat_handles_connection_error(self, _):
        agent = OllamaAgent()
        result = agent.chat("test")
        assert "error" in result.lower() or len(result) > 0  # graceful error string

    @patch("phinance.agents.ollama_agent.requests.get")
    def test_check_ollama_ready_true(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        assert check_ollama_ready() is True

    @patch("phinance.agents.ollama_agent.requests.get")
    def test_check_ollama_ready_false_on_500(self, mock_get):
        mock_get.return_value = MagicMock(status_code=500)
        assert check_ollama_ready() is False

    @patch("phinance.agents.ollama_agent.requests.get", side_effect=ConnectionError)
    def test_check_ollama_ready_false_on_error(self, _):
        assert check_ollama_ready() is False

    @patch("phinance.agents.ollama_agent.requests.get")
    def test_list_models_returns_list(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "llama3.2"}]},
        )
        models = list_ollama_models()
        assert isinstance(models, list)

    @patch("phinance.agents.ollama_agent.requests.get", side_effect=ConnectionError)
    def test_list_models_returns_empty_on_error(self, _):
        models = list_ollama_models()
        assert models == []

    def test_model_default(self):
        agent = OllamaAgent()
        assert agent.model == "llama3.2"

    def test_custom_model(self):
        agent = OllamaAgent(model="gemma2")
        assert agent.model == "gemma2"

    def test_host_stored(self):
        agent = OllamaAgent(host="http://myserver:11434")
        assert "myserver" in agent.host


# ─────────────────────────────────────────────────────────────────────────────
#  AgentOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentOrchestrator:

    def _orch(self, **kwargs) -> AgentOrchestrator:
        return AgentOrchestrator(agents=[RuleBasedAgent()], **kwargs)

    def test_returns_orchestrator_result(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert isinstance(result, OrchestratorResult)

    def test_consensus_action_valid(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert result.consensus_action in {"buy", "sell", "hold"}

    def test_consensus_conf_in_range(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert 0.0 <= result.consensus_conf <= 1.0

    def test_regime_string_non_empty(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert len(result.regime) > 0

    def test_composite_signal_in_range(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(
            ohlcv,
            indicators={"RSI": {"enabled": True, "params": {"period": 14}}},
        )
        assert -1.1 <= result.composite_signal <= 1.1

    def test_agent_results_list_non_empty(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert len(result.agent_results) == 1

    def test_summary_non_empty(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert len(result.summary) > 0

    def test_elapsed_ms_positive(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(ohlcv)
        assert result.elapsed_ms > 0

    def test_to_dict_has_expected_keys(self):
        ohlcv = _flat_ohlcv()
        result = self._orch().run(ohlcv)
        d = result.to_dict()
        assert "consensus_action" in d
        assert "regime" in d
        assert "agent_results" in d

    def test_no_indicators_zero_signal(self):
        ohlcv = _flat_ohlcv()
        result = self._orch().run(ohlcv, indicators=None)
        assert result.composite_signal == 0.0

    def test_disabled_indicator_skipped(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(
            ohlcv,
            indicators={"RSI": {"enabled": False, "params": {}}},
        )
        assert result.composite_signal == 0.0

    def test_register_adds_agent(self):
        orch = AgentOrchestrator(agents=[])
        orch.register(RuleBasedAgent())
        assert len(orch.agents) == 1

    def test_multiple_agents_aggregated(self):
        ohlcv = _trending_ohlcv()
        orch = AgentOrchestrator(agents=[RuleBasedAgent(), RuleBasedAgent()])
        result = orch.run(ohlcv)
        assert len(result.agent_results) == 2
        assert result.consensus_action in {"buy", "sell", "hold"}

    def test_backtest_fn_called(self):
        ohlcv = _flat_ohlcv()
        mock_bt = MagicMock(return_value={"sharpe": 1.2, "max_drawdown": -0.05, "total_return": 0.1, "cagr": 0.05, "trades": 10})
        orch = AgentOrchestrator(agents=[RuleBasedAgent()], backtest_fn=mock_bt)
        result = orch.run(ohlcv)
        mock_bt.assert_called_once()
        assert result.backtest_summary.get("sharpe") == 1.2

    def test_skip_unhealthy_agent(self):
        class DeadAgent(_ConcreteAgent):
            def health_check(self):
                return False

        orch = AgentOrchestrator(agents=[DeadAgent()], skip_unhealthy=True)
        result = orch.run(_flat_ohlcv())
        assert len(result.agent_results) == 0

    def test_unhealthy_agent_included_when_flag_off(self):
        class DeadAgent(_ConcreteAgent):
            def health_check(self):
                return False

        orch = AgentOrchestrator(agents=[DeadAgent()], skip_unhealthy=False)
        result = orch.run(_flat_ohlcv())
        assert len(result.agent_results) == 1

    def test_no_agents_returns_hold(self):
        orch = AgentOrchestrator(agents=[])
        result = orch.run(_flat_ohlcv())
        assert result.consensus_action == "hold"

    def test_agent_crash_does_not_abort(self):
        class CrashAgent(AgentBase):
            @property
            def name(self): return "CrashAgent"
            def analyze(self, ctx): raise RuntimeError("boom")

        orch = AgentOrchestrator(agents=[CrashAgent(), RuleBasedAgent()], skip_unhealthy=False)
        result = orch.run(_flat_ohlcv())
        assert result.consensus_action in {"buy", "sell", "hold"}

    def test_unknown_indicator_gracefully_skipped(self):
        ohlcv = _trending_ohlcv()
        result = self._orch().run(
            ohlcv,
            indicators={"NonExistentXYZ": {"enabled": True, "params": {}}},
        )
        assert result.composite_signal == 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  run_with_agents convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWithAgents:

    def test_returns_orchestrator_result(self):
        ohlcv = _trending_ohlcv()
        result = run_with_agents(ohlcv)
        assert isinstance(result, OrchestratorResult)

    def test_default_agents_rule_based(self):
        ohlcv = _trending_ohlcv()
        result = run_with_agents(ohlcv)
        assert len(result.agent_results) >= 1
        assert result.agent_results[0].agent == "RuleBasedAgent"

    def test_custom_agents(self):
        ohlcv = _flat_ohlcv()
        result = run_with_agents(ohlcv, agents=[RuleBasedAgent(buy_threshold=0.1)])
        assert result.consensus_action in {"buy", "sell", "hold"}

    def test_with_indicators(self):
        ohlcv = _trending_ohlcv()
        result = run_with_agents(
            ohlcv,
            indicators={"RSI": {"enabled": True, "params": {}},
                        "MACD": {"enabled": True, "params": {}}},
        )
        assert isinstance(result.composite_signal, float)

    def test_with_weights(self):
        ohlcv = _trending_ohlcv()
        result = run_with_agents(
            ohlcv,
            indicators={"RSI": {"enabled": True, "params": {}}},
            weights={"RSI": 1.0},
        )
        assert -1.1 <= result.composite_signal <= 1.1

    def test_consensus_in_valid_actions(self):
        ohlcv = _flat_ohlcv()
        result = run_with_agents(ohlcv)
        assert result.consensus_action in {"buy", "sell", "hold"}


# ─────────────────────────────────────────────────────────────────────────────
#  __init__ exports
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentsInit:

    def test_imports(self):
        from phinance.agents import (
            AgentBase, AgentResult, AgentCapability,
            RuleBasedAgent, OllamaAgent,
            AgentOrchestrator, OrchestratorResult, run_with_agents,
        )
        assert callable(run_with_agents)

    def test_all_exports_present(self):
        import phinance.agents as ag
        for name in ["AgentBase", "AgentResult", "RuleBasedAgent",
                     "AgentOrchestrator", "run_with_agents"]:
            assert hasattr(ag, name)

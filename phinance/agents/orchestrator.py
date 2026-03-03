"""
phinance.agents.orchestrator
==============================

Agent orchestrator: runs a backtest pipeline with optional agent oversight.

The orchestrator:
  1. Detects the current market regime from OHLCV data.
  2. Computes blended indicator signals.
  3. Runs a quick backtest.
  4. Dispatches context to all registered agents.
  5. Aggregates agent results into a final consensus recommendation.

Classes / Functions
-------------------
  AgentOrchestrator     — main orchestration class
  OrchestratorResult    — typed result of a full orchestration run
  run_with_agents       — convenience function for the full pipeline

Usage
-----
    from phinance.agents.orchestrator import AgentOrchestrator
    from phinance.agents.rule_agent   import RuleBasedAgent

    orch = AgentOrchestrator(agents=[RuleBasedAgent()])
    result = orch.run(ohlcv_df, indicators={"RSI": {"enabled": True, "params": {}}})
    print(result.consensus_action, result.summary)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from phinance.agents.base import AgentBase, AgentResult, AgentCapability
from phinance.blending.regime_detector import detect_regime
from phinance.strategies.indicator_catalog import compute_indicator
from phinance.blending.blender import blend_signals
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── OrchestratorResult ────────────────────────────────────────────────────────


@dataclass
class OrchestratorResult:
    """
    Full result of a single orchestration run.

    Attributes
    ----------
    consensus_action : str   — ``"buy"`` | ``"sell"`` | ``"hold"``
    consensus_conf   : float — weighted average confidence across all agents
    regime           : str   — detected regime label
    composite_signal : float — blended indicator signal in [-1, 1]
    agent_results    : list  — individual AgentResult objects
    backtest_summary : dict  — optional backtest stats (sharpe, drawdown, etc.)
    summary          : str   — human-readable paragraph
    elapsed_ms       : float — wall-clock time in milliseconds
    """

    consensus_action:  str
    consensus_conf:    float
    regime:            str
    composite_signal:  float
    agent_results:     List[AgentResult] = field(default_factory=list)
    backtest_summary:  Dict[str, Any]    = field(default_factory=dict)
    summary:           str               = ""
    elapsed_ms:        float             = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus_action":  self.consensus_action,
            "consensus_conf":    self.consensus_conf,
            "regime":            self.regime,
            "composite_signal":  self.composite_signal,
            "agent_results":     [r.to_dict() for r in self.agent_results],
            "backtest_summary":  self.backtest_summary,
            "summary":           self.summary,
            "elapsed_ms":        self.elapsed_ms,
        }


# ── AgentOrchestrator ─────────────────────────────────────────────────────────


class AgentOrchestrator:
    """
    Coordinates multiple agents through a shared backtest + signal pipeline.

    Parameters
    ----------
    agents           : list[AgentBase] — agents to consult
    blend_method     : str             — signal blend method (default ``"weighted_sum"``)
    backtest_fn      : callable, opt   — ``(ohlcv, **kwargs) → dict`` for backtest stats
    skip_unhealthy   : bool            — skip agents whose ``health_check()`` fails
    """

    def __init__(
        self,
        agents:         Optional[List[AgentBase]] = None,
        blend_method:   str = "weighted_sum",
        backtest_fn:    Optional[Any] = None,
        skip_unhealthy: bool = True,
    ) -> None:
        self.agents         = agents or []
        self.blend_method   = blend_method
        self.backtest_fn    = backtest_fn
        self.skip_unhealthy = skip_unhealthy

    def register(self, agent: AgentBase) -> None:
        """Add an agent to the pool."""
        self.agents.append(agent)

    # ── main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        ohlcv: pd.DataFrame,
        indicators: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        backtest_kwargs: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        Run the full agent-overseen pipeline.

        Parameters
        ----------
        ohlcv            : pd.DataFrame — OHLCV data
        indicators       : dict, opt    — ``{name: {"enabled": bool, "params": {...}}}``
        weights          : dict, opt    — ``{indicator_name: weight}`` for blending
        backtest_kwargs  : dict, opt    — extra kwargs forwarded to ``backtest_fn``

        Returns
        -------
        OrchestratorResult
        """
        t0 = time.perf_counter()

        # ── 1. Regime detection ───────────────────────────────────────────
        regime_series = detect_regime(ohlcv)
        current_regime = str(regime_series.iloc[-1]) if len(regime_series) > 0 else "RANGE"

        # ── 2. Compute indicator signals ──────────────────────────────────
        signals_dict: Dict[str, pd.Series] = {}
        if indicators:
            for name, cfg in indicators.items():
                if not cfg.get("enabled", True):
                    continue
                params = cfg.get("params", {})
                try:
                    sig = compute_indicator(name, ohlcv, params)
                    signals_dict[name] = sig
                except Exception as exc:
                    logger.warning("Orchestrator: indicator %s failed: %s", name, exc)

        # ── 3. Blend signals ──────────────────────────────────────────────
        if signals_dict:
            signals_df = pd.DataFrame(signals_dict).fillna(0.0)
            blended = blend_signals(signals_df, weights=weights or {}, method=self.blend_method)
            composite_signal = float(blended.iloc[-1]) if len(blended) > 0 else 0.0
        else:
            composite_signal = 0.0

        # ── 4. Run backtest (optional) ────────────────────────────────────
        backtest_summary: Dict[str, Any] = {}
        if self.backtest_fn is not None:
            try:
                bk = self.backtest_fn(ohlcv, **(backtest_kwargs or {}))
                if isinstance(bk, dict):
                    backtest_summary = {
                        k: bk[k]
                        for k in ("total_return", "cagr", "max_drawdown", "sharpe", "trades")
                        if k in bk
                    }
            except Exception as exc:
                logger.warning("Orchestrator: backtest_fn failed: %s", exc)

        # ── 5. Build context for agents ───────────────────────────────────
        context: Dict[str, Any] = {
            "ohlcv":        ohlcv,
            "regime":       current_regime,
            "signal":       composite_signal,
            "backtest":     backtest_summary,
            "indicators":   indicators or {},
            "signals_dict": signals_dict,
        }

        # ── 6. Dispatch to agents ─────────────────────────────────────────
        agent_results: List[AgentResult] = []
        active_agents = [
            a for a in self.agents
            if not self.skip_unhealthy or a.health_check()
        ]

        for agent in active_agents:
            try:
                result = agent.analyze(context)
                agent_results.append(result)
                logger.debug("Agent %s → %s (conf %.2f)", agent.name, result.action, result.confidence)
            except Exception as exc:
                logger.error("Agent %s raised: %s", agent.name, exc)

        # ── 7. Aggregate consensus ─────────────────────────────────────────
        consensus_action, consensus_conf = self._aggregate(agent_results)

        # ── 8. Build summary ──────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        summary = self._make_summary(
            consensus_action, consensus_conf, current_regime,
            composite_signal, agent_results, backtest_summary,
        )

        return OrchestratorResult(
            consensus_action=consensus_action,
            consensus_conf=round(consensus_conf, 4),
            regime=current_regime,
            composite_signal=round(composite_signal, 4),
            agent_results=agent_results,
            backtest_summary=backtest_summary,
            summary=summary,
            elapsed_ms=round(elapsed_ms, 1),
        )

    # ── aggregation ──────────────────────────────────────────────────────────

    def _aggregate(self, results: List[AgentResult]):
        """Confidence-weighted vote to determine consensus action."""
        if not results:
            return "hold", 0.0

        scores: Dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_w = 0.0
        for r in results:
            w = r.confidence
            action = r.action if r.action in scores else "hold"
            scores[action] += w
            total_w += w

        if total_w == 0:
            return "hold", 0.0

        consensus = max(scores, key=lambda k: scores[k])
        avg_conf = scores[consensus] / total_w if total_w > 0 else 0.0
        return consensus, min(avg_conf, 1.0)

    def _make_summary(
        self,
        action: str,
        conf: float,
        regime: str,
        signal: float,
        results: List[AgentResult],
        backtest: dict,
    ) -> str:
        lines = [
            f"Consensus: {action.upper()} (confidence {conf:.1%})",
            f"Regime: {regime} | Blended signal: {signal:+.3f}",
        ]
        if backtest:
            sh = backtest.get("sharpe")
            dd = backtest.get("max_drawdown")
            if sh is not None:
                lines.append(f"Backtest — Sharpe: {sh:.2f}, Max drawdown: {dd:.1%}" if dd else f"Backtest — Sharpe: {sh:.2f}")
        lines.append(f"Agents consulted: {len(results)}")
        for r in results:
            lines.append(f"  • {r.agent}: {r.action} ({r.confidence:.1%}) — {r.rationale}")
        return "\n".join(lines)


# ── Convenience function ──────────────────────────────────────────────────────


def run_with_agents(
    ohlcv: pd.DataFrame,
    agents: Optional[List[AgentBase]] = None,
    indicators: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    blend_method: str = "weighted_sum",
    backtest_fn: Optional[Any] = None,
) -> OrchestratorResult:
    """
    One-shot convenience wrapper around ``AgentOrchestrator.run()``.

    Parameters
    ----------
    ohlcv        : pd.DataFrame  — OHLCV data
    agents       : list, opt     — agents to use (defaults to [RuleBasedAgent()])
    indicators   : dict, opt     — active indicators config
    weights      : dict, opt     — indicator blend weights
    blend_method : str           — blend method name
    backtest_fn  : callable, opt — backtest function ``(ohlcv) → dict``

    Returns
    -------
    OrchestratorResult
    """
    from phinance.agents.rule_agent import RuleBasedAgent
    if agents is None:
        agents = [RuleBasedAgent()]
    orch = AgentOrchestrator(
        agents=agents,
        blend_method=blend_method,
        backtest_fn=backtest_fn,
    )
    return orch.run(ohlcv, indicators=indicators, weights=weights)

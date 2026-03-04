"""
phinance.agents.rule_agent
===========================

Rule-based (deterministic) agent that interprets a composite signal and
market regime to recommend a trade action.

This agent does *not* require an LLM or external service — it runs
entirely locally with zero latency.

Classes
-------
  RuleBasedAgent — fast, deterministic agent using signal + regime rules

Rules
-----
  | Signal threshold | Regime modifier | Action  |
  |-----------------|-----------------|---------|
  | signal ≥ +0.3   | any upward      | buy     |
  | signal ≤ -0.3   | any downward    | sell    |
  | else            | HIGHVOL         | hold    |
  | else            | any             | hold    |

Confidence is proportional to ``abs(signal)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from phinance.agents.base import AgentBase, AgentCapability, AgentResult


_BUY_THRESHOLD  = 0.3
_SELL_THRESHOLD = -0.3

_BULLISH_REGIMES = {"TREND_UP", "BREAKOUT_UP"}
_BEARISH_REGIMES = {"TREND_DN", "BREAKOUT_DN"}
_VOLATILE_REGIMES = {"HIGHVOL"}


class RuleBasedAgent(AgentBase):
    """
    Fast, deterministic agent — no LLM or network required.

    Parameters
    ----------
    buy_threshold  : float — signal ≥ this → buy signal (default 0.3)
    sell_threshold : float — signal ≤ this → sell signal (default -0.3)
    regime_boost   : float — multiplier added to confidence in regime-aligned situations
    """

    def __init__(
        self,
        buy_threshold:  float = _BUY_THRESHOLD,
        sell_threshold: float = _SELL_THRESHOLD,
        regime_boost:   float = 0.15,
    ) -> None:
        self._buy_threshold  = buy_threshold
        self._sell_threshold = sell_threshold
        self._regime_boost   = regime_boost

    @property
    def name(self) -> str:
        return "RuleBasedAgent"

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.TRADE_SIGNAL,
            AgentCapability.RISK_ASSESSMENT,
            AgentCapability.BACKTEST_OVERSIGHT,
        ]

    # ── core analyze ─────────────────────────────────────────────────────────

    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        """
        Determine action from composite signal and regime label.

        Expected context keys
        ---------------------
        signal  : float — composite blended signal in [-1, 1]
        regime  : str   — market regime label (e.g. "TREND_UP")
        backtest: dict  — optional backtest summary (sharpe, max_drawdown, etc.)
        """
        signal  = float(context.get("signal", 0.0))
        regime  = str(context.get("regime", "RANGE"))
        backtest = context.get("backtest", {})

        action, confidence, rationale = self._decide(signal, regime, backtest)

        return AgentResult(
            agent=self.name,
            action=action,
            confidence=confidence,
            rationale=rationale,
            data={
                "signal":  signal,
                "regime":  regime,
                "backtest_sharpe": backtest.get("sharpe"),
            },
        )

    # ── internal ─────────────────────────────────────────────────────────────

    def _decide(
        self,
        signal:   float,
        regime:   str,
        backtest: dict,
    ):
        abs_sig = abs(signal)
        base_conf = min(abs_sig, 1.0)

        # Risk gate: if recent Sharpe is very negative, dampen confidence
        sharpe = backtest.get("sharpe", None)
        if sharpe is not None and sharpe < -1.0:
            base_conf *= 0.5

        # Regime-alignment boost
        if signal >= self._buy_threshold:
            boost = self._regime_boost if regime in _BULLISH_REGIMES else 0.0
            conf = min(base_conf + boost, 1.0)
            rationale = (
                f"Signal={signal:.3f} (≥ {self._buy_threshold}) in regime {regime}. "
                f"Regime {'aligns ✓' if regime in _BULLISH_REGIMES else 'neutral'}."
            )
            return "buy", round(conf, 4), rationale

        if signal <= self._sell_threshold:
            boost = self._regime_boost if regime in _BEARISH_REGIMES else 0.0
            conf = min(base_conf + boost, 1.0)
            rationale = (
                f"Signal={signal:.3f} (≤ {self._sell_threshold}) in regime {regime}. "
                f"Regime {'aligns ✓' if regime in _BEARISH_REGIMES else 'neutral'}."
            )
            return "sell", round(conf, 4), rationale

        # Neutral zone
        risk_note = ""
        if regime in _VOLATILE_REGIMES:
            risk_note = " High-vol regime detected — holding."
        elif regime in _BULLISH_REGIMES:
            risk_note = " Bullish regime but signal too weak to enter."
        conf = min(base_conf + 0.2, 1.0)  # moderate confidence in hold
        rationale = (
            f"Signal={signal:.3f} within neutral band [{self._sell_threshold}, "
            f"{self._buy_threshold}].{risk_note}"
        )
        return "hold", round(conf, 4), rationale

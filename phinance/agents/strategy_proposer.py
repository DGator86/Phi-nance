"""
phinance.agents.strategy_proposer
====================================

StrategyProposerAgent — autonomously generates indicator combinations
and blend-weight proposals based on current market regime and recent
backtest performance.

The agent performs:
  1. Regime detection on the supplied OHLCV data.
  2. Catalogue-wide scoring of every registered indicator using a fast
     directional-accuracy probe (no full backtest).
  3. Selects the top-N indicators that are statistically compatible
     with the current regime.
  4. Proposes blend weights using an inverse-volatility heuristic over
     the indicator signals.
  5. Returns a ``StrategyProposal`` dataclass ready for validation and
     deployment.

Public API
----------
  StrategyProposal        — typed proposal dataclass
  StrategyProposerAgent   — AgentBase subclass
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phinance.agents.base import AgentBase, AgentResult, AgentCapability
from phinance.blending.regime_detector import detect_regime
from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
from phinance.optimization.evaluators import direction_accuracy
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

# Indicators known to work well in each regime
_REGIME_AFFINITY: Dict[str, List[str]] = {
    "TREND_UP":     ["EMA Cross", "MACD", "DEMA", "TEMA", "HMA", "KAMA", "Ichimoku", "Aroon"],
    "TREND_DN":     ["EMA Cross", "MACD", "DEMA", "TEMA", "RSI", "Williams %R", "Ichimoku"],
    "RANGE":        ["RSI", "Bollinger", "Stochastic", "Williams %R", "CCI", "Mean Reversion",
                     "Keltner", "Donchian"],
    "HIGHVOL":      ["ATR", "Bollinger", "Keltner", "Ulcer Index", "Elder Ray", "Mass Index"],
    "LOWVOL":       ["MACD", "EMA Cross", "Dual SMA", "VWAP", "VWMA", "ZLEMA"],
    "BREAKOUT_UP":  ["Breakout", "Donchian", "Mass Index", "TRIX", "KST", "OBV"],
    "BREAKOUT_DN":  ["Breakout", "Donchian", "RSI", "TRIX", "DPO", "Elder Ray"],
    "UNKNOWN":      ["RSI", "MACD", "Bollinger", "EMA Cross", "ATR"],
}


# ── StrategyProposal ──────────────────────────────────────────────────────────


@dataclass
class StrategyProposal:
    """
    A proposed trading strategy ready for validation and deployment.

    Attributes
    ----------
    indicators   : dict  — ``{name: {"enabled": True, "params": {...}}}``
    weights      : dict  — ``{name: float}`` blend weights (sum to 1)
    blend_method : str   — ``"weighted_sum"`` | ``"regime_weighted"`` | …
    regime       : str   — regime that triggered this proposal
    scores       : dict  — ``{name: direction_accuracy_score}``
    rationale    : str   — human-readable explanation
    created_at   : float — epoch timestamp
    """

    indicators:   Dict[str, Any]
    weights:      Dict[str, float]
    blend_method: str   = "weighted_sum"
    regime:       str   = "UNKNOWN"
    scores:       Dict[str, float] = field(default_factory=dict)
    rationale:    str   = ""
    created_at:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicators":   self.indicators,
            "weights":      self.weights,
            "blend_method": self.blend_method,
            "regime":       self.regime,
            "scores":       self.scores,
            "rationale":    self.rationale,
            "created_at":   self.created_at,
        }


# ── StrategyProposerAgent ─────────────────────────────────────────────────────


class StrategyProposerAgent(AgentBase):
    """
    Autonomously proposes indicator combinations for the current regime.

    Parameters
    ----------
    top_n          : int  — number of indicators to include in proposal (default 4)
    min_score      : float — minimum direction_accuracy to include indicator (default 0.50)
    blend_method   : str  — blending method for the proposal (default ``"weighted_sum"``)
    probe_bars     : int  — bars to use for the quick accuracy probe (default 60)
    """

    capabilities = [
        AgentCapability.MARKET_ANALYSIS,
        AgentCapability.TRADE_SIGNAL,
        AgentCapability.PARAMETER_TUNING,
    ]

    def __init__(
        self,
        top_n:        int   = 4,
        min_score:    float = 0.50,
        blend_method: str   = "weighted_sum",
        probe_bars:   int   = 60,
    ) -> None:
        super().__init__()
        self.top_n        = top_n
        self.min_score    = min_score
        self.blend_method = blend_method
        self.probe_bars   = probe_bars

    @property
    def name(self) -> str:
        return "StrategyProposerAgent"

    # ── Core analysis ─────────────────────────────────────────────────────────

    def analyze(self, context: dict) -> AgentResult:
        """Generate an action + rationale from context dict.

        Expects ``context["ohlcv"]`` — a pd.DataFrame with OHLCV columns.
        """
        ohlcv = context.get("ohlcv")
        if ohlcv is None or len(ohlcv) < 30:
            return AgentResult(
                agent=self.name,
                action="hold",
                confidence=0.0,
                rationale="Insufficient OHLCV data for proposal.",
            )

        proposal = self.propose(ohlcv)
        action   = "hold"
        if len(proposal.indicators) > 0:
            avg_score = float(np.mean(list(proposal.scores.values()))) if proposal.scores else 0.5
            if avg_score > 0.55:
                action = "buy" if "TREND_UP" in proposal.regime or "BREAKOUT_UP" in proposal.regime else "hold"
            if avg_score < 0.45:
                action = "sell"

        return AgentResult(
            agent=self.name,
            action=action,
            confidence=float(np.mean(list(proposal.scores.values()))) if proposal.scores else 0.5,
            rationale=proposal.rationale,
            data={"proposal": proposal.to_dict()},
        )

    def propose(self, ohlcv: pd.DataFrame) -> StrategyProposal:
        """
        Build a ``StrategyProposal`` for the given OHLCV data.

        Parameters
        ----------
        ohlcv : pd.DataFrame — OHLCV data (at least 30 bars)

        Returns
        -------
        StrategyProposal
        """
        t0 = time.time()

        # 1. Detect regime
        regime_series = detect_regime(ohlcv)
        regime = str(regime_series.iloc[-1]) if len(regime_series) > 0 else "UNKNOWN"

        # 2. Get regime-compatible candidates
        candidates = _REGIME_AFFINITY.get(regime, _REGIME_AFFINITY["UNKNOWN"])
        # Only include candidates that are registered in the catalog
        candidates = [c for c in candidates if c in INDICATOR_CATALOG]

        # 3. Probe each candidate with direction accuracy
        probe_df = ohlcv.tail(max(self.probe_bars, 30))
        scores: Dict[str, float] = {}
        for ind_name in candidates:
            try:
                score = direction_accuracy(probe_df, ind_name, {})
                if score >= self.min_score:
                    scores[ind_name] = float(score)
            except Exception as exc:
                logger.debug("Probe failed for %s: %s", ind_name, exc)

        # 4. Select top-N
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        selected = {name: score for name, score in top}

        if not selected:
            # Fallback to regime affinity list regardless of score
            fallback = candidates[: self.top_n]
            selected = {n: 0.5 for n in fallback}

        # 5. Build indicators dict
        indicators: Dict[str, Any] = {
            name: {"enabled": True, "params": {}}
            for name in selected
        }

        # 6. Inverse-score weighting (better scores → higher weight)
        total_score = sum(selected.values()) or 1.0
        weights = {name: s / total_score for name, s in selected.items()}

        elapsed = (time.time() - t0) * 1000
        rationale = (
            f"Regime: {regime} | "
            f"Top indicators: {list(selected.keys())} | "
            f"Avg accuracy: {np.mean(list(selected.values())):.3f} | "
            f"Elapsed: {elapsed:.1f}ms"
        )

        logger.info("Proposal: %s", rationale)

        return StrategyProposal(
            indicators=indicators,
            weights=weights,
            blend_method=self.blend_method,
            regime=regime,
            scores=selected,
            rationale=rationale,
        )

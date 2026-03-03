"""
phinance.agents.base
=====================

Abstract base class and protocol definitions for Phi-nance AI agents.

All agents must implement the ``AgentBase`` interface to ensure
plug-and-play composability within the orchestrator.

Classes
-------
  AgentBase         — Abstract base class for all agents
  AgentResult       — Typed dataclass for agent responses
  AgentCapability   — Enum of capabilities an agent can declare

Usage
-----
    from phinance.agents.base import AgentBase, AgentResult

    class MyAgent(AgentBase):
        @property
        def name(self) -> str:
            return "MyAgent"

        def analyze(self, context: dict) -> AgentResult:
            return AgentResult(
                agent=self.name,
                action="hold",
                confidence=0.6,
                rationale="Neutral regime detected.",
            )
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── AgentCapability ────────────────────────────────────────────────────────────


class AgentCapability(str, Enum):
    """Capabilities an agent can declare to the orchestrator."""
    MARKET_ANALYSIS     = "market_analysis"
    TRADE_SIGNAL        = "trade_signal"
    RISK_ASSESSMENT     = "risk_assessment"
    REGIME_DETECTION    = "regime_detection"
    PARAMETER_TUNING    = "parameter_tuning"
    BACKTEST_OVERSIGHT  = "backtest_oversight"
    NATURAL_LANGUAGE    = "natural_language"


# ── AgentResult ────────────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """
    Typed result container returned by every agent.

    Attributes
    ----------
    agent       : str   — name of the agent that produced this result
    action      : str   — recommended action: ``"buy"`` | ``"sell"`` | ``"hold"`` | ``"none"``
    confidence  : float — confidence in the action, 0.0–1.0
    rationale   : str   — human-readable explanation
    data        : dict  — optional structured payload (e.g. Greeks, regime, params)
    timestamp   : float — epoch time of result generation
    """

    agent:      str
    action:     str                         # "buy" | "sell" | "hold" | "none"
    confidence: float                       # 0.0 – 1.0
    rationale:  str
    data:       Dict[str, Any] = field(default_factory=dict)
    timestamp:  float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        valid_actions = {"buy", "sell", "hold", "none", "unknown"}
        if self.action not in valid_actions:
            raise ValueError(
                f"AgentResult.action must be one of {valid_actions}, "
                f"got {self.action!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"AgentResult.confidence must be in [0, 1], "
                f"got {self.confidence}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent":      self.agent,
            "action":     self.action,
            "confidence": self.confidence,
            "rationale":  self.rationale,
            "data":       self.data,
            "timestamp":  self.timestamp,
        }

    @property
    def signal_value(self) -> float:
        """Map action → numeric signal in [-1, 1] weighted by confidence."""
        mapping = {"buy": 1.0, "sell": -1.0, "hold": 0.0, "none": 0.0, "unknown": 0.0}
        return mapping.get(self.action, 0.0) * self.confidence


# ── AgentBase ──────────────────────────────────────────────────────────────────


class AgentBase(ABC):
    """
    Abstract base for all Phi-nance agents.

    Subclasses must implement:
      - ``name``    property
      - ``analyze`` method

    Optionally override:
      - ``capabilities`` — advertise what the agent can do
      - ``health_check`` — return True if backend is reachable
    """

    # ── identity ────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable agent name."""
        ...

    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent supports.  Override in subclass."""
        return [AgentCapability.MARKET_ANALYSIS]

    # ── core interface ───────────────────────────────────────────────────────

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> AgentResult:
        """
        Analyse the given context and return an AgentResult.

        Parameters
        ----------
        context : dict
            Arbitrary key-value context.  The orchestrator passes at minimum:
              - ``"ohlcv"``       pd.DataFrame — OHLCV data
              - ``"regime"``      str          — current market regime label
              - ``"signal"``      float        — blended composite signal in [-1, 1]
              - ``"backtest"``    dict         — latest backtest result summary
              - ``"indicators"``  dict         — active indicator states

        Returns
        -------
        AgentResult
        """
        ...

    def health_check(self) -> bool:
        """Return True when the agent's backend is available.  Default True."""
        return True

    # ── helpers ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        caps = ", ".join(c.value for c in self.capabilities)
        return f"{self.__class__.__name__}(name={self.name!r}, capabilities=[{caps}])"

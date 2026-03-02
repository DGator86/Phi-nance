"""
Options AI Advisor — Regime-aware AI agent for the options pipeline.
====================================================================
Wraps OllamaAgent (or PlutusAdvisor when available) to provide AI-driven
recommendations specifically for the options pipeline:

  1. Strategy selection  — which options structure for current regime/IV/GEX
  2. Entry timing        — should we enter now or wait?
  3. Position sizing     — how much capital to allocate given regime clarity
  4. Risk assessment     — key risks for the recommended structure

The advisor receives the same regime_probs, gamma_features, and IV data
that OptionsEngine uses, but augments the rule-based logic with LLM
reasoning for edge cases and nuanced regime transitions.

Graceful degradation: when Ollama is unavailable the advisor falls back
to a deterministic rule-based engine (no dead ends).

Usage
-----
    >>> from phi.options.ai_advisor import OptionsAIAdvisor
    >>> advisor = OptionsAIAdvisor()
    >>> rec = advisor.recommend(
    ...     symbol="SPY", spot=450.0, hist_vol=0.22,
    ...     regime_probs={"TREND_UP": 0.6, "RANGE": 0.3, "TREND_DN": 0.1},
    ...     gamma_features={"gamma_net": 0.15, "gex_flip_zone": 0},
    ...     iv_regime="HIGH_IV",
    ... )
    >>> print(rec.structure, rec.confidence, rec.reasoning)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsRecommendation:
    """Structured recommendation from the Options AI Advisor."""
    structure:      str              # e.g. 'iron_condor', 'bull_call_spread'
    level:          str              # L1 / L2 / L3
    direction:      str              # BULLISH / BEARISH / NEUTRAL / VOLATILE
    confidence:     float            # 0.0 – 1.0
    reasoning:      str              # human-readable explanation
    risk_note:      str              # key risk for this position
    position_size:  float            # recommended fraction of capital (0.0 – 1.0)
    entry_signal:   str              # ENTER / WAIT / SKIP
    regime:         str              # dominant regime used
    vol_regime:     str              # HIGH_IV / NORMAL / LOW_IV
    gex_regime:     str              # PINNING / NEUTRAL / AMPLIFY / FLIP
    source:         str              # 'ai' or 'rules' (fallback)
    raw_llm_output: Optional[str] = None

    def is_actionable(self, min_confidence: float = 0.40) -> bool:
        return self.entry_signal == "ENTER" and self.confidence >= min_confidence

    def summary(self) -> str:
        return (
            f"[{self.level}] {self.structure.upper()} | {self.direction} | "
            f"Conf={self.confidence:.0%} | Signal={self.entry_signal} | "
            f"Size={self.position_size:.0%} | Source={self.source}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Regime → structure mapping (deterministic fallback)
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_MAP: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    # (regime, vol_regime) → (structure, level, direction)
    "TREND_UP": {
        "LOW_IV":  ("long_call",        "L1", "BULLISH"),
        "NORMAL":  ("bull_call_spread",  "L1", "BULLISH"),
        "HIGH_IV": ("bull_put_spread",   "L1", "BULLISH"),
    },
    "TREND_DN": {
        "LOW_IV":  ("long_put",          "L1", "BEARISH"),
        "NORMAL":  ("bear_put_spread",   "L1", "BEARISH"),
        "HIGH_IV": ("bear_call_spread",  "L1", "BEARISH"),
    },
    "RANGE": {
        "LOW_IV":  ("covered_call",      "L2", "NEUTRAL"),
        "NORMAL":  ("iron_condor",       "L3", "NEUTRAL"),
        "HIGH_IV": ("iron_condor",       "L3", "NEUTRAL"),
    },
    "BREAKOUT_UP": {
        "LOW_IV":  ("long_straddle",     "L3", "VOLATILE"),
        "NORMAL":  ("long_call",         "L1", "BULLISH"),
        "HIGH_IV": ("bull_put_spread",   "L1", "BULLISH"),
    },
    "BREAKOUT_DN": {
        "LOW_IV":  ("long_straddle",     "L3", "VOLATILE"),
        "NORMAL":  ("long_put",          "L1", "BEARISH"),
        "HIGH_IV": ("bear_call_spread",  "L1", "BEARISH"),
    },
    "HIGHVOL": {
        "LOW_IV":  ("long_strangle",     "L3", "VOLATILE"),
        "NORMAL":  ("bear_put_spread",   "L1", "BEARISH"),
        "HIGH_IV": ("collar",            "L2", "BEARISH"),
    },
    "LOWVOL": {
        "LOW_IV":  ("calendar_spread",   "L3", "NEUTRAL"),
        "NORMAL":  ("iron_butterfly",    "L3", "NEUTRAL"),
        "HIGH_IV": ("iron_condor",       "L3", "NEUTRAL"),
    },
}

_SYSTEM_PROMPT = """\
You are an expert quantitative options strategist in the Phi-nance trading system.
You receive structured market state data and must recommend an options strategy.

Return ONLY valid JSON (no markdown, no commentary):
{
  "structure": "<strategy_name>",
  "level": "L1|L2|L3",
  "direction": "BULLISH|BEARISH|NEUTRAL|VOLATILE",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one concise sentence>",
  "risk_note": "<key risk in one sentence>",
  "position_size": <float 0.01-0.20>,
  "entry_signal": "ENTER|WAIT|SKIP"
}

Valid structures: long_call, long_put, bull_call_spread, bear_put_spread,
bull_put_spread, bear_call_spread, long_straddle, long_strangle,
short_straddle, iron_condor, iron_butterfly, calendar_spread,
covered_call, collar.

Rules:
1. HIGH_IV regimes favor credit (sell) strategies; LOW_IV favors debit (buy).
2. PINNING GEX → range-bound strategies; AMPLIFY → breakout structures.
3. GEX FLIP zone → elevated caution, reduce confidence and position_size.
4. Use high confidence (>0.7) only when regime + vol + GEX all agree.
5. SKIP when signals conflict; WAIT when regime is transitioning.
6. Position size: 5-10% normal, 2-5% uncertain, 10-15% high conviction.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Options AI Advisor
# ──────────────────────────────────────────────────────────────────────────────

class OptionsAIAdvisor:
    """
    AI-powered options strategy advisor.

    Attempts Ollama LLM for nuanced reasoning; falls back to deterministic
    rules when Ollama is unavailable. Never returns None — always produces
    a complete OptionsRecommendation.

    Parameters
    ----------
    model       : Ollama model name (default: llama3.2)
    host        : Ollama server URL
    timeout     : LLM request timeout in seconds
    use_ollama  : attempt LLM recommendations (True) or force rules (False)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 60,
        use_ollama: bool = True,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self.use_ollama = use_ollama
        self._ollama_agent = None

    def _get_agent(self):
        """Lazy-load OllamaAgent."""
        if self._ollama_agent is None:
            try:
                from phi.agents import OllamaAgent
                self._ollama_agent = OllamaAgent(
                    model=self.model,
                    host=self.host,
                    timeout=self.timeout,
                )
            except Exception as exc:
                logger.warning("OptionsAIAdvisor: cannot load OllamaAgent: %s", exc)
        return self._ollama_agent

    def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        if not self.use_ollama:
            return False
        try:
            from phi.agents import check_ollama_ready
            return check_ollama_ready(self.host)
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        symbol: str,
        spot: float,
        hist_vol: float,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
        iv_regime: str = "NORMAL",
        gex_regime: Optional[str] = None,
        chain_summary: Optional[Dict[str, float]] = None,
    ) -> OptionsRecommendation:
        """
        Generate a complete options strategy recommendation.

        Always returns a valid OptionsRecommendation — never None.
        Tries LLM first, falls back to deterministic rules.

        Parameters
        ----------
        symbol          : ticker (e.g. 'SPY')
        spot            : current underlying price
        hist_vol        : annualised realised volatility (e.g. 0.22)
        regime_probs    : {regime_name: probability} from RegimeEngine
        gamma_features  : {gamma_net, gamma_wall_distance, ...} from GammaSurface
        iv_regime       : 'HIGH_IV', 'NORMAL', or 'LOW_IV'
        gex_regime      : 'PINNING', 'NEUTRAL', 'AMPLIFY', 'FLIP' (auto-derived if None)
        chain_summary   : optional IV surface features (iv_atm, iv_skew, ...)
        """
        # Derive GEX regime if not provided
        if gex_regime is None:
            gex_regime = self._classify_gex(gamma_features)

        dominant = self._dominant_regime(regime_probs)
        confidence_base = self._base_confidence(regime_probs, gamma_features)

        # Try LLM
        if self.use_ollama and self.is_available():
            try:
                rec = self._recommend_llm(
                    symbol, spot, hist_vol, regime_probs,
                    gamma_features, iv_regime, gex_regime,
                    dominant, chain_summary,
                )
                if rec is not None:
                    return rec
            except Exception as exc:
                logger.warning("OptionsAIAdvisor LLM failed: %s — falling back to rules", exc)

        # Deterministic fallback
        return self._recommend_rules(
            symbol, spot, hist_vol, regime_probs,
            gamma_features, iv_regime, gex_regime,
            dominant, confidence_base,
        )

    # ──────────────────────────────────────────────────────────────────────
    # LLM recommendation
    # ──────────────────────────────────────────────────────────────────────

    def _recommend_llm(
        self,
        symbol: str,
        spot: float,
        hist_vol: float,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
        iv_regime: str,
        gex_regime: str,
        dominant: str,
        chain_summary: Optional[Dict[str, float]],
    ) -> Optional[OptionsRecommendation]:
        """Ask the LLM for a structured recommendation."""
        agent = self._get_agent()
        if agent is None:
            return None

        # Build market state prompt
        prompt_data = {
            "symbol": symbol,
            "spot_price": round(spot, 2),
            "hist_vol_ann": round(hist_vol, 4),
            "iv_regime": iv_regime,
            "gex_regime": gex_regime,
            "dominant_regime": dominant,
            "regime_probs": {k: round(v, 3) for k, v in regime_probs.items()},
            "gamma_features": {k: round(v, 4) for k, v in gamma_features.items()},
        }
        if chain_summary:
            prompt_data["iv_surface"] = {k: round(v, 4) for k, v in chain_summary.items()}

        prompt = (
            f"Market state for {symbol}:\n"
            f"```json\n{json.dumps(prompt_data, indent=2)}\n```\n\n"
            "Based on this market state, recommend an options strategy. "
            "Return ONLY the JSON object."
        )

        reply = agent.chat(prompt, system=_SYSTEM_PROMPT)
        if not reply or "[Ollama error" in reply:
            return None

        # Parse JSON from reply
        parsed = self._parse_llm_json(reply)
        if parsed is None:
            logger.warning("OptionsAIAdvisor: could not parse LLM JSON: %s", reply[:200])
            return None

        # Validate and build recommendation
        structure = str(parsed.get("structure", "iron_condor")).lower().strip()
        valid_structures = {
            "long_call", "long_put", "bull_call_spread", "bear_put_spread",
            "bull_put_spread", "bear_call_spread", "long_straddle",
            "long_strangle", "short_straddle", "iron_condor",
            "iron_butterfly", "calendar_spread", "covered_call", "collar",
        }
        if structure not in valid_structures:
            structure = "iron_condor"

        level = str(parsed.get("level", "L3")).upper()
        if level not in ("L1", "L2", "L3"):
            level = "L3"

        direction = str(parsed.get("direction", "NEUTRAL")).upper()
        if direction not in ("BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"):
            direction = "NEUTRAL"

        confidence = float(np.clip(parsed.get("confidence", 0.5), 0.0, 1.0))
        position_size = float(np.clip(parsed.get("position_size", 0.05), 0.01, 0.20))

        entry_signal = str(parsed.get("entry_signal", "ENTER")).upper()
        if entry_signal not in ("ENTER", "WAIT", "SKIP"):
            entry_signal = "ENTER"

        return OptionsRecommendation(
            structure=structure,
            level=level,
            direction=direction,
            confidence=confidence,
            reasoning=str(parsed.get("reasoning", "AI recommendation based on current market state.")),
            risk_note=str(parsed.get("risk_note", "Monitor regime transitions and IV changes.")),
            position_size=position_size,
            entry_signal=entry_signal,
            regime=dominant,
            vol_regime=iv_regime,
            gex_regime=gex_regime,
            source="ai",
            raw_llm_output=reply,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Deterministic fallback
    # ──────────────────────────────────────────────────────────────────────

    def _recommend_rules(
        self,
        symbol: str,
        spot: float,
        hist_vol: float,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
        iv_regime: str,
        gex_regime: str,
        dominant: str,
        confidence_base: float,
    ) -> OptionsRecommendation:
        """Deterministic rule-based recommendation — never fails."""

        # Look up structure from fallback map
        regime_map = _FALLBACK_MAP.get(dominant, _FALLBACK_MAP["RANGE"])
        structure, level, direction = regime_map.get(
            iv_regime, regime_map.get("NORMAL", ("iron_condor", "L3", "NEUTRAL"))
        )

        # GEX overrides
        if gex_regime == "FLIP":
            # Unstable — prefer straddle, reduce confidence
            structure = "long_straddle"
            level = "L3"
            direction = "VOLATILE"
            confidence_base *= 0.6

        elif gex_regime == "AMPLIFY" and dominant in ("RANGE", "LOWVOL"):
            # Expect breakout from range
            structure = "long_strangle"
            level = "L3"
            direction = "VOLATILE"

        elif gex_regime == "PINNING" and dominant in ("TREND_UP", "TREND_DN"):
            # Pinning resists trend — reduce confidence
            confidence_base *= 0.75

        # Entry signal
        if confidence_base >= 0.50:
            entry_signal = "ENTER"
        elif confidence_base >= 0.35:
            entry_signal = "WAIT"
        else:
            entry_signal = "SKIP"

        # Position sizing based on confidence
        if confidence_base >= 0.70:
            position_size = 0.12
        elif confidence_base >= 0.50:
            position_size = 0.08
        elif confidence_base >= 0.35:
            position_size = 0.05
        else:
            position_size = 0.03

        # Build reasoning
        reasoning = self._build_reasoning(dominant, iv_regime, gex_regime, structure)
        risk_note = self._build_risk_note(structure, iv_regime, gex_regime)

        return OptionsRecommendation(
            structure=structure,
            level=level,
            direction=direction,
            confidence=round(float(np.clip(confidence_base, 0.0, 1.0)), 3),
            reasoning=reasoning,
            risk_note=risk_note,
            position_size=position_size,
            entry_signal=entry_signal,
            regime=dominant,
            vol_regime=iv_regime,
            gex_regime=gex_regime,
            source="rules",
        )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_regime(regime_probs: Dict[str, float]) -> str:
        if not regime_probs:
            return "RANGE"
        return max(regime_probs, key=lambda k: regime_probs[k])

    @staticmethod
    def _classify_gex(gamma_features: Dict[str, float]) -> str:
        if gamma_features.get("gex_flip_zone", 0) >= 0.5:
            return "FLIP"
        gn = float(gamma_features.get("gamma_net", 0.0))
        if gn >= 0.30:
            return "PINNING"
        if gn <= -0.30:
            return "AMPLIFY"
        return "NEUTRAL"

    @staticmethod
    def _base_confidence(
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
    ) -> float:
        """Compute base confidence from regime clarity and GEX stability."""
        probs = np.array(list(regime_probs.values()), dtype=float)
        if len(probs) == 0:
            return 0.3
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum()
        max_entropy = math.log(len(probs) + 1e-10)
        entropy = -float((probs * np.log(probs)).sum())
        c_regime = float(np.clip(1.0 - entropy / (max_entropy + 1e-10), 0.0, 1.0))

        # GEX stability
        c_gex = 1.0 - float(gamma_features.get("gex_flip_zone", 0.0)) * 0.5

        return float(np.clip(c_regime * c_gex, 0.0, 1.0))

    @staticmethod
    def _build_reasoning(
        dominant: str, iv_regime: str, gex_regime: str, structure: str,
    ) -> str:
        regime_desc = {
            "TREND_UP": "uptrending",
            "TREND_DN": "downtrending",
            "RANGE": "range-bound",
            "BREAKOUT_UP": "breaking out upward",
            "BREAKOUT_DN": "breaking out downward",
            "HIGHVOL": "high-volatility",
            "LOWVOL": "low-volatility",
        }
        vol_desc = {
            "HIGH_IV": "implied volatility elevated vs realised (sell premium)",
            "LOW_IV": "implied volatility depressed vs realised (buy premium)",
            "NORMAL": "IV near fair value",
        }
        gex_desc = {
            "PINNING": "dealer gamma positive (range-pinning expected)",
            "AMPLIFY": "dealer gamma negative (moves may amplify)",
            "NEUTRAL": "balanced dealer positioning",
            "FLIP": "near GEX zero-crossing (elevated instability)",
        }

        r = regime_desc.get(dominant, dominant)
        v = vol_desc.get(iv_regime, iv_regime)
        g = gex_desc.get(gex_regime, gex_regime)

        return f"Market is {r} with {v}; {g} — {structure.replace('_', ' ')} selected."

    @staticmethod
    def _build_risk_note(structure: str, iv_regime: str, gex_regime: str) -> str:
        risks = {
            "long_call": "Time decay accelerates near expiry; total premium at risk.",
            "long_put": "Time decay accelerates near expiry; total premium at risk.",
            "bull_call_spread": "Capped upside; max loss is net debit paid.",
            "bear_put_spread": "Capped downside gain; max loss is net debit paid.",
            "bull_put_spread": "Unlimited loss below lower strike; early assignment risk.",
            "bear_call_spread": "Unlimited loss above upper strike; early assignment risk.",
            "long_straddle": "Requires significant move; double premium at risk if range-bound.",
            "long_strangle": "Requires even larger move than straddle; full premium at risk.",
            "short_straddle": "Unlimited risk both directions; margin-intensive.",
            "iron_condor": "Max loss if underlying breaches either wing; watch for breakouts.",
            "iron_butterfly": "Concentrated max loss zone; requires tight range.",
            "calendar_spread": "Sensitive to IV term structure changes; front-month pin risk.",
            "covered_call": "Capped upside; underlying still fully exposed to downside.",
            "collar": "Capped both ways; opportunity cost in strong rallies.",
        }
        base = risks.get(structure, "Monitor position Greeks and regime changes.")

        if gex_regime == "FLIP":
            base += " GEX flip zone adds regime instability risk."
        if iv_regime == "HIGH_IV" and structure.startswith("long_"):
            base += " High IV means elevated premium cost — consider timing."

        return base

    @staticmethod
    def _parse_llm_json(text: str) -> Optional[Dict]:
        """Extract the first JSON object from LLM text."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Find first { ... } block
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        return None
        return None

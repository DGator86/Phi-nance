"""Pydantic state containers for the force-field strategy engine."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RegimeTensor(BaseModel):
    """Multi-layer regime state; each family is a sub-regime probability map (simplex)."""

    indicator: dict[str, float]
    liquidity: dict[str, float]
    greek: dict[str, float]
    news: dict[str, float]
    industry: dict[str, float]
    model_config = {"frozen": False}

    def flattened(self, prefix: tuple[str, str, str, str, str] | None = None) -> dict[str, float]:
        pref = prefix or ("ind", "liq", "gr", "news", "indus")
        flat: dict[str, float] = {}
        flat.update({f"{pref[0]}.{k}": float(v) for k, v in self.indicator.items()})
        flat.update({f"{pref[1]}.{k}": float(v) for k, v in self.liquidity.items()})
        flat.update({f"{pref[2]}.{k}": float(v) for k, v in self.greek.items()})
        flat.update({f"{pref[3]}.{k}": float(v) for k, v in self.news.items()})
        flat.update({f"{pref[4]}.{k}": float(v) for k, v in self.industry.items()})
        return flat


class ForceVector(BaseModel):
    directional: float = 0.0
    magnet: float = 0.0
    expansion: float = 0.0
    friction: float = 0.0
    systemic: float = 0.0


class MassState(BaseModel):
    raw_mass: float
    normalized_mass: float
    cap_score: float = 0.0
    liquidity_score: float = 0.0
    gamma_stability_score: float = 0.0
    event_sensitivity_score: float = 0.0


class MotionState(BaseModel):
    expected_direction: float = 0.0
    expected_expansion: float = 0.0
    pinning_strength: float = 0.0
    execution_difficulty: float = 0.0
    systemic_coupling: float = 0.0


class StrategyScoreResult(BaseModel):
    strategy_name: str
    score: float
    confidence: float = 1.0
    rationale: list[str] = Field(default_factory=list)
    reject_reasons: list[str] = Field(default_factory=list)


class EngineState(BaseModel):
    """Full per-bar state: tensor → force → mass → motion."""

    regime: RegimeTensor
    force: ForceVector
    mass: MassState
    motion: MotionState
    features: dict[str, Any] = Field(default_factory=dict)

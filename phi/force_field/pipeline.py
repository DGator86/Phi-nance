"""End-to-end: features → regime tensor → force → mass → motion → ranked strategies."""

from __future__ import annotations

from typing import Mapping

from phi.force_field.acceleration import derive_motion
from phi.force_field.force_mapper import ForceMapper
from phi.force_field.mass_model import MassModel
from phi.force_field.regime_engines import build_regime_tensor_from_features
from phi.force_field.schemas import EngineState, RegimeTensor, StrategyScoreResult
from phi.force_field.scorer import score_strategy
from phi.force_field.strategy_profiles import STRATEGY_PROFILES
from phi.force_field.vetoes import VetoEngine


def run_engine(
    features: Mapping[str, float],
    *,
    regime: RegimeTensor | None = None,
    force_mapper: ForceMapper | None = None,
    mass_model: MassModel | None = None,
) -> EngineState:
    R = regime or build_regime_tensor_from_features(features)
    mapper = force_mapper or ForceMapper()
    mass_m = mass_model or MassModel()
    F = mapper.map(R)
    m = mass_m.compute(features, R)
    motion = derive_motion(F, m)
    return EngineState(regime=R, force=F, mass=m, motion=motion, features=dict(features))


def rank_strategies(
    state: EngineState,
    *,
    vetoes: VetoEngine | None = None,
    execution_penalty: float = 0.0,
) -> list[StrategyScoreResult]:
    v = vetoes or VetoEngine()
    out: list[StrategyScoreResult] = []
    for name, profile in STRATEGY_PROFILES.items():
        reject = v.check(name, state.regime, state.motion, state.features)
        if reject:
            out.append(
                StrategyScoreResult(
                    strategy_name=name,
                    score=-999.0,
                    confidence=0.0,
                    reject_reasons=reject,
                )
            )
            continue
        sc = score_strategy(
            profile,
            state.motion,
            state.regime,
            state.features,
            execution_penalty=execution_penalty,
        )
        out.append(StrategyScoreResult(strategy_name=name, score=sc, confidence=1.0, reject_reasons=[]))
    return sorted(out, key=lambda r: r.score, reverse=True)


def run_pipeline(
    features: Mapping[str, float],
    *,
    regime: RegimeTensor | None = None,
    force_mapper: ForceMapper | None = None,
    mass_model: MassModel | None = None,
    vetoes: VetoEngine | None = None,
    execution_penalty: float = 0.0,
) -> tuple[EngineState, list[StrategyScoreResult]]:
    state = run_engine(
        features,
        regime=regime,
        force_mapper=force_mapper,
        mass_model=mass_model,
    )
    ranked = rank_strategies(state, vetoes=vetoes, execution_penalty=execution_penalty)
    return state, ranked

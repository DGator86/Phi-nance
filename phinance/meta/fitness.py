"""Fitness evaluation for GP-discovered strategies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from deap import gp

from phinance.backtest.vectorized import run_vectorized_backtest
from phinance.meta.primitives import PrimitiveContext, build_feature_frame


@dataclass
class FitnessResult:
    sharpe: float
    max_drawdown: float
    total_return: float
    num_trades: int
    fitness: float


def _hash_individual(individual: gp.PrimitiveTree) -> str:
    return hashlib.sha256(str(individual).encode("utf-8")).hexdigest()


class GPFitnessEvaluator:
    """Evaluate individuals by compiling expression trees into trading signals."""

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        primitive_context: PrimitiveContext,
        signal_clip: float = 1.0,
        signal_threshold: float = 0.1,
        penalty_turnover: float = 0.0,
    ) -> None:
        self.ohlcv = ohlcv
        self.context = primitive_context
        self.signal_clip = float(signal_clip)
        self.signal_threshold = float(signal_threshold)
        self.penalty_turnover = float(penalty_turnover)
        self.cache: Dict[str, FitnessResult] = {}

        features = build_feature_frame(ohlcv)
        self.features = features[self.context.feature_names]
        self.feature_arrays = [self.features[name].to_numpy(dtype=float) for name in self.context.feature_names]

    def evaluate(self, individual: gp.PrimitiveTree, toolbox: Any) -> tuple[float]:
        key = _hash_individual(individual)
        cached = self.cache.get(key)
        if cached is not None:
            return (cached.fitness,)

        func = toolbox.compile(expr=individual)
        raw_signal = np.asarray(func(*self.feature_arrays), dtype=float)
        if raw_signal.ndim == 0:
            raw_signal = np.full(len(self.ohlcv), float(raw_signal), dtype=float)
        signal = np.clip(np.nan_to_num(raw_signal, nan=0.0, posinf=0.0, neginf=0.0), -self.signal_clip, self.signal_clip)
        signal_series = pd.Series(signal, index=self.ohlcv.index, dtype=float)

        backtest = run_vectorized_backtest(
            self.ohlcv,
            signal=signal_series,
            signal_threshold=self.signal_threshold,
            position_style="long_short",
            symbol="GP",
        )
        turnover = float(np.mean(np.abs(np.diff(backtest.positions)))) if len(backtest.positions) > 1 else 0.0

        fitness = float(backtest.sharpe - 0.5 * max(backtest.max_drawdown, 0.0) - self.penalty_turnover * turnover)
        result = FitnessResult(
            sharpe=float(backtest.sharpe),
            max_drawdown=float(backtest.max_drawdown),
            total_return=float(backtest.total_return),
            num_trades=int(backtest.num_trades),
            fitness=fitness,
        )
        self.cache[key] = result

        individual.meta_metrics = {
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "total_return": result.total_return,
            "num_trades": result.num_trades,
            "turnover": turnover,
        }
        return (fitness,)

    def get_cached_metrics(self, individual: gp.PrimitiveTree) -> Dict[str, float]:
        key = _hash_individual(individual)
        result = self.cache.get(key)
        if result is None:
            return {}
        return {
            "sharpe": result.sharpe,
            "max_drawdown": result.max_drawdown,
            "total_return": result.total_return,
            "num_trades": float(result.num_trades),
            "fitness": result.fitness,
        }


def build_fitness_evaluator(
    ohlcv: pd.DataFrame,
    primitive_context: PrimitiveContext,
    distributed_runner: Any | None = None,
    **kwargs: Any,
) -> "GPFitnessEvaluator":
    """Return distributed evaluator when a runner is supplied, else local evaluator."""
    if distributed_runner is None:
        return GPFitnessEvaluator(ohlcv=ohlcv, primitive_context=primitive_context, **kwargs)

    from phinance.meta.distributed_fitness import DistributedGPFitnessEvaluator

    return DistributedGPFitnessEvaluator(
        ohlcv=ohlcv,
        primitive_context=primitive_context,
        runner=distributed_runner,
        **kwargs,
    )

"""Distributed fitness evaluation for GP populations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from deap import gp

from phinance.backtest.distributed_runner import DistributedBacktestRunner
from phinance.meta.fitness import FitnessResult, GPFitnessEvaluator, _hash_individual


class DistributedGPFitnessEvaluator(GPFitnessEvaluator):
    """Batch-evaluate individuals using DistributedBacktestRunner."""

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        primitive_context: Any,
        runner: DistributedBacktestRunner,
        signal_clip: float = 1.0,
        signal_threshold: float = 0.1,
        penalty_turnover: float = 0.0,
    ) -> None:
        super().__init__(
            ohlcv=ohlcv,
            primitive_context=primitive_context,
            signal_clip=signal_clip,
            signal_threshold=signal_threshold,
            penalty_turnover=penalty_turnover,
        )
        self.runner = runner

    def evaluate_population(self, individuals: Iterable[gp.PrimitiveTree], toolbox: Any) -> List[Tuple[float]]:
        individuals_list = list(individuals)
        pending: List[gp.PrimitiveTree] = []
        configs: List[Dict[str, Any]] = []

        for individual in individuals_list:
            key = _hash_individual(individual)
            cached = self.cache.get(key)
            if cached is not None:
                individual.fitness.values = (cached.fitness,)
                continue

            func = toolbox.compile(expr=individual)
            raw_signal = np.asarray(func(*self.feature_arrays), dtype=float)
            if raw_signal.ndim == 0:
                raw_signal = np.full(len(self.ohlcv), float(raw_signal), dtype=float)

            signal = np.clip(
                np.nan_to_num(raw_signal, nan=0.0, posinf=0.0, neginf=0.0),
                -self.signal_clip,
                self.signal_clip,
            )
            signal_series = pd.Series(signal, index=self.ohlcv.index, dtype=float)
            turnover = float(np.mean(np.abs(np.diff(signal)))) if len(signal) > 1 else 0.0

            pending.append(individual)
            configs.append(
                {
                    "engine": "vectorized",
                    "ohlcv": self.ohlcv,
                    "signal": signal_series,
                    "symbol": "GP",
                    "signal_threshold": self.signal_threshold,
                    "position_style": "long_short",
                    "_turnover": turnover,
                }
            )

        if configs:
            outputs = self.runner.run_parallel(configs)
            for individual, cfg, output in zip(pending, configs, outputs):
                if output["status"] != "ok" or output["result"] is None:
                    individual.fitness.values = (-1e9,)
                    continue

                result = output["result"]
                turnover = float(cfg.get("_turnover", 0.0))
                fitness = float(
                    float(result.get("sharpe_ratio", 0.0))
                    - 0.5 * max(float(result.get("max_drawdown", 0.0)), 0.0)
                    - self.penalty_turnover * turnover
                )

                cached = FitnessResult(
                    sharpe=float(result.get("sharpe_ratio", 0.0)),
                    max_drawdown=float(result.get("max_drawdown", 0.0)),
                    total_return=float(result.get("total_return", 0.0)),
                    num_trades=int(result.get("num_trades", 0)),
                    fitness=fitness,
                )
                self.cache[_hash_individual(individual)] = cached

                individual.meta_metrics = {
                    "sharpe": cached.sharpe,
                    "max_drawdown": cached.max_drawdown,
                    "total_return": cached.total_return,
                    "num_trades": cached.num_trades,
                    "turnover": turnover,
                }
                individual.fitness.values = (fitness,)

        return [tuple(ind.fitness.values) for ind in individuals_list]

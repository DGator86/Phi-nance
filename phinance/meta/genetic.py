"""Genetic programming core for strategy discovery."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from phinance.backtest.distributed_runner import DistributedBacktestRunner
from phinance.meta.fitness import build_fitness_evaluator
from phinance.meta.primitives import build_feature_frame, build_primitive_set


@dataclass
class GPConfig:
    population_size: int = 50
    generations: int = 20
    tournament_size: int = 3
    cxpb: float = 0.5
    mutpb: float = 0.3
    min_depth: int = 1
    max_depth: int = 4
    top_k: int = 5
    random_seed: int | None = None
    distributed_enabled: bool = False
    distributed_num_cpus: int | None = None
    distributed_address: str | None = None
    distributed_use_ray: bool = True
    distributed_timeout_s: float | None = None


class GeneticStrategySearch:
    """Run GP evolution over expression trees that output trading signals."""

    def __init__(self, ohlcv: pd.DataFrame, config: GPConfig | None = None) -> None:
        self.ohlcv = ohlcv
        self.config = config or GPConfig()
        self.rng = random.Random(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        features = build_feature_frame(ohlcv)
        pset, context = build_primitive_set(features.columns.tolist())
        self.pset = pset
        self.context = context

        if not hasattr(creator, "FitnessMaxMeta"):
            creator.create("FitnessMaxMeta", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualMeta"):
            creator.create("IndividualMeta", gp.PrimitiveTree, fitness=creator.FitnessMaxMeta)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.config.min_depth, max_=self.config.max_depth)
        toolbox.register("individual", tools.initIterate, creator.IndividualMeta, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        self.distributed_runner = DistributedBacktestRunner(
            enabled=self.config.distributed_enabled,
            use_ray=self.config.distributed_use_ray,
            num_cpus=self.config.distributed_num_cpus,
            address=self.config.distributed_address,
            timeout_s=self.config.distributed_timeout_s,
        )
        runner = self.distributed_runner if self.distributed_runner.is_distributed else None
        self.evaluator = build_fitness_evaluator(ohlcv=ohlcv, primitive_context=self.context, distributed_runner=runner)

        toolbox.register("evaluate", self.evaluator.evaluate, toolbox=toolbox)
        toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=64))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=64))

        self.toolbox = toolbox

    def _evaluate_invalid(self, individuals: List[gp.PrimitiveTree]) -> None:
        if not individuals:
            return

        if hasattr(self.evaluator, "evaluate_population"):
            fits = self.evaluator.evaluate_population(individuals, self.toolbox)
        else:
            fits = self.toolbox.map(self.toolbox.evaluate, individuals)

        for ind, fit in zip(individuals, fits):
            ind.fitness.values = fit

    def evolve(self) -> Dict[str, Any]:
        pop = self.toolbox.population(n=self.config.population_size)
        hall_of_fame = tools.HallOfFame(maxsize=self.config.top_k)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        history: List[Dict[str, float]] = []

        invalid = [ind for ind in pop if not ind.fitness.valid]
        self._evaluate_invalid(invalid)

        for gen in range(self.config.generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if self.rng.random() < self.config.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if self.rng.random() < self.config.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            self._evaluate_invalid(invalid)

            pop[:] = offspring
            hall_of_fame.update(pop)
            record = stats.compile(pop)
            history.append({"generation": float(gen), **{k: float(v) for k, v in record.items()}})

        best = []
        for rank, individual in enumerate(hall_of_fame):
            best.append(
                {
                    "strategy_id": str(uuid.uuid4())[:8],
                    "rank": rank,
                    "expression": str(individual),
                    "fitness": float(individual.fitness.values[0]),
                    "metrics": self.evaluator.get_cached_metrics(individual),
                }
            )

        self.distributed_runner.shutdown()
        return {"history": history, "best_strategies": best, "feature_names": self.context.feature_names}

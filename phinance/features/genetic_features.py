"""Genetic programming feature discovery for predictive expressions."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools


def _safe_div(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    return np.divide(a, np.where(np.abs(b) < 1e-8, 1e-8, b))


def _safe_log(x: np.ndarray | float) -> np.ndarray | float:
    return np.log(np.abs(x) + 1e-8)


def _safe_sqrt(x: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt(np.abs(x))


@dataclass
class GPFeatureConfig:
    population_size: int = 24
    generations: int = 5
    top_k: int = 5
    tournament_size: int = 3
    crossover_prob: float = 0.5
    mutation_prob: float = 0.3
    min_depth: int = 1
    max_depth: int = 3
    random_seed: int = 7


class GPFeatureDiscovery:
    """Evolve symbolic expressions predictive of next-period returns."""

    def __init__(self, data: pd.DataFrame, target_col: str = "target", config: GPFeatureConfig | None = None) -> None:
        self.data = data.copy()
        self.target_col = target_col
        self.config = config or GPFeatureConfig()

        if target_col not in self.data.columns:
            self.data[target_col] = self.data["close"].pct_change().shift(-1).fillna(0.0)

        self.feature_columns = [
            col for col in self.data.columns if col != target_col and np.issubdtype(self.data[col].dtype, np.number)
        ]
        self.pset = self._build_pset(self.feature_columns)
        self.toolbox = self._build_toolbox()

    def _build_pset(self, feature_columns: list[str]) -> gp.PrimitiveSet:
        pset = gp.PrimitiveSet("MAIN", len(feature_columns))
        pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(feature_columns)})
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(_safe_div, 2)
        pset.addPrimitive(np.sin, 1)
        pset.addPrimitive(np.cos, 1)
        pset.addPrimitive(np.tanh, 1)
        pset.addPrimitive(_safe_log, 1)
        pset.addPrimitive(_safe_sqrt, 1)
        pset.addEphemeralConstant("rand", lambda: np.random.uniform(-1.0, 1.0))
        return pset

    def _build_toolbox(self) -> base.Toolbox:
        fitness_name = "FitnessMaxFeature"
        individual_name = "IndividualFeature"
        if not hasattr(creator, fitness_name):
            creator.create(fitness_name, base.Fitness, weights=(1.0,))
        if not hasattr(creator, individual_name):
            creator.create(individual_name, gp.PrimitiveTree, fitness=getattr(creator, fitness_name))

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=self.config.min_depth, max_=self.config.max_depth)
        toolbox.register("individual", tools.initIterate, getattr(creator, individual_name), toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)
        toolbox.register("evaluate", self._evaluate_individual)
        toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=64))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=64))
        return toolbox

    def _evaluate_individual(self, individual: gp.PrimitiveTree) -> tuple[float]:
        func = self.toolbox.compile(expr=individual)
        arrays = [self.data[name].to_numpy(dtype=float) for name in self.feature_columns]
        y = self.data[self.target_col].to_numpy(dtype=float)
        try:
            values = np.asarray(func(*arrays), dtype=float)
        except Exception:  # noqa: BLE001
            return (-1.0,)

        if values.shape != y.shape:
            values = np.broadcast_to(values, y.shape)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(values) < 1e-8 or np.std(y) < 1e-8:
            return (0.0,)

        corr = np.corrcoef(values, y)[0, 1]
        if math.isnan(corr):
            corr = 0.0
        return (float(abs(corr)),)

    def evolve(self) -> list[dict[str, Any]]:
        np.random.seed(self.config.random_seed)
        pop = self.toolbox.population(n=self.config.population_size)
        hof = tools.HallOfFame(maxsize=self.config.top_k)

        for individual in pop:
            individual.fitness.values = self.toolbox.evaluate(individual)

        for _ in range(self.config.generations):
            offspring = list(map(self.toolbox.clone, self.toolbox.select(pop, len(pop))))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < self.config.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < self.config.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)
            pop[:] = offspring
            hof.update(pop)

        results: list[dict[str, Any]] = []
        for ind in hof:
            results.append(
                {
                    "expression": str(ind),
                    "fitness": float(ind.fitness.values[0]),
                }
            )
        return results


def evaluate_expression(expression: str, frame: pd.DataFrame) -> float:
    """Evaluate expression string against the latest row in a dataframe."""
    local_vars = {col: float(frame[col].iloc[-1]) for col in frame.columns if np.issubdtype(frame[col].dtype, np.number)}
    local_vars.update({"sin": math.sin, "cos": math.cos, "tanh": math.tanh, "log": lambda x: math.log(abs(x) + 1e-8), "sqrt": lambda x: math.sqrt(abs(x))})
    local_vars.update({"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "_safe_div": lambda a, b: a / (b if abs(b) > 1e-8 else 1e-8)})
    return float(eval(expression, {"__builtins__": {}}, local_vars))  # noqa: S307

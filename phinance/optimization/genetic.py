"""
phinance.optimization.genetic
================================

Genetic Algorithm (GA) hyperparameter optimisation — a pure-NumPy
implementation that requires no extra dependencies.

GA applies natural-selection metaphors to parameter tuning:
  • **Population** — N candidate parameter sets (chromosomes)
  • **Fitness**     — objective_fn score (higher = better)
  • **Selection**   — tournament selection chooses parents
  • **Crossover**   — uniform crossover combines two parent chromosomes
  • **Mutation**    — randomly changes a gene with probability ``mutation_rate``
  • **Elitism**     — top-k individuals are preserved unchanged each generation

This tends to find diverse solutions and escape local minima better than
pure random search, especially when the search space is large and
multi-modal.

References
----------
* Holland (1975) — "Adaptation in Natural and Artificial Systems"
* Goldberg (1989) — "Genetic Algorithms in Search, Optimization, and ML"
* Mitchell (1998) — "An Introduction to Genetic Algorithms"

Public API
----------
  genetic_search(ohlcv, objective_fn, param_grid, population_size,
                 n_generations, mutation_rate, crossover_rate, elitism_k,
                 tournament_k, seed) → (best_params, best_score)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _encode_individual(
    rng: np.random.Generator,
    param_grid: Dict[str, list],
) -> Dict[str, Any]:
    """Sample a random individual (chromosome) from the parameter grid."""
    return {k: vlist[int(rng.integers(len(vlist)))] for k, vlist in param_grid.items()}


def _tournament_select(
    population: List[Dict[str, Any]],
    fitness: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Tournament selection: pick k random candidates, return the fittest."""
    indices = rng.integers(0, len(population), size=k)
    best_idx = indices[np.argmax(fitness[indices])]
    return population[best_idx]


def _uniform_crossover(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Uniform crossover: each gene inherited independently from either parent."""
    return {
        k: (parent_a[k] if rng.random() < 0.5 else parent_b[k])
        for k in parent_a
    }


def _mutate(
    individual: Dict[str, Any],
    param_grid: Dict[str, list],
    mutation_rate: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Point mutation: each gene replaced with a random grid value at p=mutation_rate."""
    return {
        k: (rng.choice(param_grid[k]) if rng.random() < mutation_rate else v)
        for k, v in individual.items()
    }


def _evaluate_population(
    population: List[Dict[str, Any]],
    ohlcv: pd.DataFrame,
    objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
) -> np.ndarray:
    """Evaluate fitness for every individual in the population."""
    fitness = np.zeros(len(population))
    for i, ind in enumerate(population):
        try:
            score = float(objective_fn(ohlcv, ind))
            fitness[i] = score if np.isfinite(score) else 0.0
        except Exception:
            fitness[i] = 0.0
    return fitness


# ── Public API ────────────────────────────────────────────────────────────────


def genetic_search(
    ohlcv: pd.DataFrame,
    objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
    param_grid: Dict[str, list],
    population_size: int = 30,
    n_generations: int = 20,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.80,
    elitism_k: int = 3,
    tournament_k: int = 4,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, Any], float]:
    """Genetic algorithm hyperparameter search.

    Parameters
    ----------
    ohlcv           : pd.DataFrame — OHLCV data
    objective_fn    : callable     — ``(ohlcv, params) → score``
    param_grid      : dict         — ``{param: [values]}``
    population_size : int          — individuals per generation (default 30)
    n_generations   : int          — number of generations (default 20)
    mutation_rate   : float        — per-gene mutation probability (default 0.15)
    crossover_rate  : float        — probability of crossover vs clone (default 0.80)
    elitism_k       : int          — top-k cloned directly to next generation (default 3)
    tournament_k    : int          — tournament selection pool size (default 4)
    seed            : int, optional — RNG seed (default 42)

    Returns
    -------
    (best_params, best_score)

    Notes
    -----
    Total evaluations ≈ ``population_size × n_generations`` minus elite clones.
    With defaults: 30 × 20 = 600 evaluations.  Set ``population_size=20,
    n_generations=10`` for a 200-evaluation budget.
    """
    if not param_grid:
        return {}, 0.0

    rng = np.random.default_rng(seed)

    # ── Initialise population ─────────────────────────────────────────────────
    population: List[Dict[str, Any]] = [
        _encode_individual(rng, param_grid) for _ in range(population_size)
    ]

    best_params: Dict[str, Any] = population[0]
    best_score = -np.inf

    for gen in range(n_generations):
        fitness = _evaluate_population(population, ohlcv, objective_fn)

        # Update global best
        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_score:
            best_score = float(fitness[gen_best_idx])
            best_params = dict(population[gen_best_idx])

        logger.debug(
            "GA gen %d/%d  best_so_far=%.4f  gen_best=%.4f",
            gen + 1,
            n_generations,
            best_score,
            float(fitness[gen_best_idx]),
        )

        # ── Elitism: carry top-k directly ─────────────────────────────────────
        elite_k = min(elitism_k, population_size)
        elite_indices = np.argsort(fitness)[-elite_k:]
        elites = [dict(population[i]) for i in elite_indices]

        # ── Breed next generation ─────────────────────────────────────────────
        next_gen: List[Dict[str, Any]] = elites[:]

        while len(next_gen) < population_size:
            parent_a = _tournament_select(population, fitness, tournament_k, rng)
            if rng.random() < crossover_rate:
                parent_b = _tournament_select(population, fitness, tournament_k, rng)
                child = _uniform_crossover(parent_a, parent_b, rng)
            else:
                child = dict(parent_a)

            child = _mutate(child, param_grid, mutation_rate, rng)
            next_gen.append(child)

        population = next_gen

    logger.info(
        "Genetic search: %d gens × %d pop, best_score=%.4f, params=%s",
        n_generations,
        population_size,
        best_score,
        best_params,
    )
    return best_params, best_score

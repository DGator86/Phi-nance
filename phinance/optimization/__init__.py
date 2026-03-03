"""
phinance.optimization — PhiAI auto-tuning and parameter search.

Sub-modules
-----------
  grid_search  — Grid and random search utilities + unified ``search()`` dispatcher
  bayesian     — Bayesian optimisation (Optuna TPE sampler)
  genetic      — Genetic Algorithm (pure-NumPy, no extra deps)
  evaluators   — Fast objective-function implementations
  explainer    — Human-readable explanation of optimisation results
  phiai        — PhiAI orchestrator (run_phiai_optimization + PhiAI class)

Search methods
--------------
  ``"random"``   — Random search (fast, good baseline)
  ``"bayesian"`` — Optuna TPE (most sample-efficient, requires optuna)
  ``"genetic"``  — Genetic Algorithm (global search, no extra deps)
  ``"grid"``     — Exhaustive grid search (tiny grids only)

Public API
----------
    from phinance.optimization import (
        run_phiai_optimization, PhiAI,
        grid_search, random_search,
        bayesian_search, genetic_search,
        search, SEARCH_METHODS,
        direction_accuracy,
    )
"""

from phinance.optimization.phiai import PhiAI, run_phiai_optimization, SEARCH_METHODS
from phinance.optimization.grid_search import grid_search, random_search, search
from phinance.optimization.bayesian import bayesian_search
from phinance.optimization.genetic import genetic_search
from phinance.optimization.evaluators import direction_accuracy

__all__ = [
    "PhiAI",
    "run_phiai_optimization",
    "SEARCH_METHODS",
    "grid_search",
    "random_search",
    "search",
    "bayesian_search",
    "genetic_search",
    "direction_accuracy",
]

"""
phinance.optimization — PhiAI auto-tuning and parameter search.

Sub-modules
-----------
  grid_search  — Grid and random search utilities
  evaluators   — Fast objective-function implementations
  explainer    — Human-readable explanation of optimisation results
  phiai        — PhiAI orchestrator (run_phiai_optimization + PhiAI class)

Public API
----------
    from phinance.optimization import run_phiai_optimization, PhiAI
"""

from phinance.optimization.phiai import PhiAI, run_phiai_optimization
from phinance.optimization.grid_search import grid_search, random_search
from phinance.optimization.evaluators import direction_accuracy

__all__ = [
    "PhiAI",
    "run_phiai_optimization",
    "grid_search",
    "random_search",
    "direction_accuracy",
]

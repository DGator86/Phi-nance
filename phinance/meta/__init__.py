"""Meta-learning package for strategy discovery."""

from phinance.meta.genetic import GPConfig, GeneticStrategySearch
from phinance.meta.search import run_meta_search

__all__ = ["GPConfig", "GeneticStrategySearch", "run_meta_search"]

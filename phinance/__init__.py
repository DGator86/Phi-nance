"""
Phi-nance — Open-Source Quantitative Research Platform
=======================================================

Version 1.1.0 — Phase 6: Walk-Forward Optimisation, Monte Carlo Simulation,
                          Feature Engineering, Robust Indicators.

Package layout
--------------
  phinance.data          — Data fetching, caching, vendors, feature engineering
  phinance.strategies    — 15 indicator strategies and catalog
  phinance.blending      — Signal blending engine (4 methods)
  phinance.optimization  — PhiAI auto-tuning, grid/random/hill-climbing search,
                           walk-forward optimisation (NEW in v1.1)
  phinance.backtest      — Backtest engine, metrics, Monte Carlo (NEW in v1.1)
  phinance.options       — Options pricing, Greeks, backtest, IV surface
  phinance.config        — RunConfig, settings, schemas
  phinance.storage       — Run history and local I/O
  phinance.utils         — Logging, timing, decorators
  phinance.agents        — LLM agents (Ollama)
  phinance.phibot        — Post-backtest review engine
  phinance.exceptions    — Custom exception hierarchy

Typical workflow (data → signal → blend → optimise → backtest → results)
-------------------------------------------------------------------------
  1. Acquire data          : phinance.data.fetch_and_cache(...)
  2. Build feature matrix  : phinance.data.features.build_feature_matrix(...)
  3. Define strategy       : phinance.strategies.indicator_catalog
  4. Blend signals         : phinance.blending.blend_signals(...)
  5. Optimise params       : phinance.optimization.run_phiai_optimization(...)
  6. Walk-forward test     : phinance.optimization.walk_forward_optimize(...)
  7. Run backtest          : phinance.backtest.runner.run_backtest(...)
  8. Monte Carlo analysis  : phinance.backtest.monte_carlo.run_monte_carlo(...)
  9. Analyse & store       : phinance.storage.run_history.RunHistory(...)
 10. Post-backtest review  : phinance.phibot.reviewer.PhiBot(...)
"""

__version__ = "1.1.0"
__author__  = "Phi-nance Contributors"
__license__ = "MIT"

# ── Convenience re-exports ────────────────────────────────────────────────────
# Callers can do:
#   from phinance import (
#       fetch_and_cache, blend_signals, RunConfig, RunHistory,
#       run_backtest, run_monte_carlo, walk_forward_optimize,
#       build_feature_matrix,
#   )

from phinance.data.cache          import fetch_and_cache, get_cached_dataset, list_cached_datasets
from phinance.data.features       import build_feature_matrix, drop_warmup, feature_names
from phinance.blending.blender    import blend_signals
from phinance.config.run_config   import RunConfig
from phinance.storage.run_history import RunHistory
from phinance.backtest.runner     import run_backtest
from phinance.backtest.monte_carlo import run_monte_carlo
from phinance.optimization.walk_forward import walk_forward_optimize
from phinance.strategies          import list_indicators, compute_indicator, INDICATOR_CATALOG

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # data
    "fetch_and_cache",
    "get_cached_dataset",
    "list_cached_datasets",
    "build_feature_matrix",
    "drop_warmup",
    "feature_names",
    # blending
    "blend_signals",
    # config
    "RunConfig",
    # storage
    "RunHistory",
    # backtest
    "run_backtest",
    "run_monte_carlo",
    # optimisation
    "walk_forward_optimize",
    # strategies
    "list_indicators",
    "compute_indicator",
    "INDICATOR_CATALOG",
]

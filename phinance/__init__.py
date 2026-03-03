"""
Phi-nance — Open-Source Quantitative Research Platform
=======================================================

Version 1.0.0

Package layout
--------------
  phinance.data          — Data fetching, caching, vendors, feature engineering
  phinance.strategies    — 15 indicator strategies and catalog
  phinance.blending      — Signal blending engine (4 methods)
  phinance.optimization  — PhiAI auto-tuning, grid/random search, walk-forward
  phinance.backtest      — Backtest engine, metrics, models, Monte Carlo
  phinance.options       — Options pricing, Greeks, backtest
  phinance.config        — RunConfig, settings, schemas
  phinance.storage       — Run history and local I/O
  phinance.utils         — Logging, timing, decorators
  phinance.agents        — LLM agents (Ollama)
  phinance.phibot        — Post-backtest review engine
  phinance.exceptions    — Custom exception hierarchy

Typical workflow (data → signal → blend → optimise → backtest → results)
-------------------------------------------------------------------------
  1. Acquire data     : phinance.data.fetch_and_cache(...)
  2. Define strategy  : phinance.strategies.indicator_catalog
  3. Blend signals    : phinance.blending.blend_signals(...)
  4. Optimise params  : phinance.optimization.run_phiai_optimization(...)
  5. Run backtest     : phinance.backtest.run_backtest(...)
  6. Analyse & store  : phinance.storage.RunHistory(...)
  7. Post-run review  : phinance.phibot.reviewer.review_backtest(...)
"""

__version__ = "1.0.0"
__author__  = "Phi-nance Contributors"
__license__ = "MIT"

# ── Convenience re-exports ────────────────────────────────────────────────────
# Callers can do:
#   from phinance import (
#       fetch_and_cache, blend_signals, RunConfig, RunHistory, run_backtest,
#   )

from phinance.data.cache          import fetch_and_cache, get_cached_dataset, list_cached_datasets
from phinance.blending.blender    import blend_signals
from phinance.config.run_config   import RunConfig
from phinance.storage.run_history import RunHistory
from phinance.backtest.runner     import run_backtest
from phinance.strategies          import list_indicators, compute_indicator, INDICATOR_CATALOG

# ── Phase 6 / extended features (import lazily to avoid hard dependencies) ───
try:
    from phinance.data.features            import build_feature_matrix, drop_warmup, feature_names
    from phinance.backtest.monte_carlo     import run_monte_carlo
    from phinance.optimization.walk_forward import walk_forward_optimize
    _PHASE6_AVAILABLE = True
except ImportError:
    _PHASE6_AVAILABLE = False
    build_feature_matrix = None  # type: ignore[assignment]
    drop_warmup          = None  # type: ignore[assignment]
    feature_names        = None  # type: ignore[assignment]
    run_monte_carlo      = None  # type: ignore[assignment]
    walk_forward_optimize = None  # type: ignore[assignment]


__all__ = [
    "__version__",
    "__author__",
    "__license__",
    # data
    "fetch_and_cache",
    "get_cached_dataset",
    "list_cached_datasets",
    # blending
    "blend_signals",
    # config
    "RunConfig",
    # storage
    "RunHistory",
    # backtest
    "run_backtest",
    # strategies
    "list_indicators",
    "compute_indicator",
    "INDICATOR_CATALOG",
    # phase 6 (may be None if dependencies unavailable)
    "build_feature_matrix",
    "drop_warmup",
    "feature_names",
    "run_monte_carlo",
    "walk_forward_optimize",
]

"""
phinance.optimization.phiai
============================

PhiAI orchestrator — auto-tuning for all active indicators.

Runs parallel random search (``ThreadPoolExecutor``) over each indicator's
parameter grid, selecting the parameter set that maximises
directional-accuracy on the supplied OHLCV dataset.

Public API
----------
  PhiAI                  — Configuration class with ``explain()`` method
  run_phiai_optimization — High-level entry point
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from phinance.optimization.grid_search import random_search
from phinance.optimization.evaluators import direction_accuracy
from phinance.optimization.explainer import build_explanation
from phinance.strategies.params import get_param_grid
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_PARALLEL = 4


# ── PhiAI class ───────────────────────────────────────────────────────────────


class PhiAI:
    """PhiAI configuration and explanation container.

    Parameters
    ----------
    max_indicators : int   — maximum indicators to enable (default 5)
    allow_shorts   : bool  — permit short-selling (default False)
    risk_cap       : float — max portfolio risk fraction (e.g. 0.02 = 2%)
    """

    def __init__(
        self,
        max_indicators: int = 5,
        allow_shorts: bool = False,
        risk_cap: Optional[float] = None,
    ) -> None:
        self.max_indicators = max_indicators
        self.allow_shorts = allow_shorts
        self.risk_cap = risk_cap
        self.changes: List[Dict[str, str]] = []

    def explain(self) -> str:
        """Return a human-readable summary of all PhiAI adjustments."""
        return build_explanation(
            self.changes,
            config={
                "max_indicators": self.max_indicators,
                "allow_shorts":   self.allow_shorts,
                "risk_cap":       self.risk_cap,
            },
        )

    def __repr__(self) -> str:
        return (
            f"PhiAI(max_indicators={self.max_indicators}, "
            f"allow_shorts={self.allow_shorts}, risk_cap={self.risk_cap})"
        )


# ── run_phiai_optimization ────────────────────────────────────────────────────


def run_phiai_optimization(
    ohlcv: pd.DataFrame,
    indicators: Dict[str, Dict[str, Any]],
    max_iter_per_indicator: int = 20,
    timeframe: str = "1D",
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Auto-tune parameters for every active indicator.

    Uses concurrent random search (up to ``_MAX_PARALLEL`` workers).
    The objective is directional accuracy on the supplied OHLCV data.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV used as the optimisation dataset.
    indicators : dict
        ``{name: {"enabled": bool, "auto_tune": bool, "params": {...}}}``
    max_iter_per_indicator : int
        Max random-search evaluations per indicator.
    timeframe : str
        Used to select the appropriate parameter grid
        (daily vs intraday periods).

    Returns
    -------
    (optimized_indicators, explanation_str)
    """
    from phinance.strategies.indicator_catalog import INDICATOR_CATALOG

    optimized: Dict[str, Dict[str, Any]] = {}
    changes: List[Dict[str, str]] = []

    def _tune_one(
        item: Tuple[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Optional[Tuple[Dict, float]]]:
        name, cfg = item
        if name not in INDICATOR_CATALOG:
            return name, cfg, None
        grid = get_param_grid(name, timeframe)
        if not grid:
            return name, cfg, None

        def obj_fn(df: pd.DataFrame, params: Dict) -> float:
            return direction_accuracy(df, name, params)

        best_params, best_score = random_search(
            ohlcv, obj_fn, grid, max_iter=max_iter_per_indicator
        )
        return name, cfg, (best_params, best_score)

    workers = min(max(len(indicators), 1), _MAX_PARALLEL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_tune_one, indicators.items()))

    for name, cfg, result in results:
        if result is None or not result[0]:
            optimized[name] = cfg
        else:
            best_params, best_score = result
            optimized[name] = {
                "enabled":    True,
                "auto_tune":  False,
                "params":     best_params,
            }
            changes.append({
                "what":   f"{name} params",
                "reason": f"Optimized → directional acc {best_score:.1%}",
            })
            logger.info(
                "PhiAI: %s → %s (acc %.1f%%)",
                name, best_params, best_score * 100,
            )

    explanation = build_explanation(changes)
    return optimized, explanation

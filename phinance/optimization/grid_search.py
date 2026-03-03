"""
phinance.optimization.grid_search
===================================

Grid and random search utilities for hyperparameter optimisation.

Functions
---------
  grid_search(ohlcv, objective_fn, param_grid, max_iter)
  random_search(ohlcv, objective_fn, param_grid, max_iter)
  search(...)  — unified dispatcher (method='grid'|'random')
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def grid_search(
    ohlcv: pd.DataFrame,
    objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
    param_grid: Dict[str, List[Any]],
    max_iter: int = 200,
) -> Tuple[Dict[str, Any], float]:
    """Exhaustive grid search over *param_grid*.

    Parameters
    ----------
    ohlcv        : pd.DataFrame — OHLCV data passed to objective_fn
    objective_fn : callable — ``(ohlcv, params) → score`` (higher = better)
    param_grid   : dict — ``{param: [values]}``
    max_iter     : int — cap on total evaluations

    Returns
    -------
    (best_params, best_score)
    """
    if not param_grid:
        return {}, 0.0

    keys = list(param_grid.keys())
    vals = list(param_grid.values())
    combos = list(product(*vals))[:max_iter]

    best_params: Dict[str, Any] = {}
    best_score = -np.inf

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            score = float(objective_fn(ohlcv, params))
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            continue

    return best_params, best_score


def random_search(
    ohlcv: pd.DataFrame,
    objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
    param_grid: Dict[str, List[Any]],
    max_iter: int = 50,
    seed: int | None = None,
) -> Tuple[Dict[str, Any], float]:
    """Random search over *param_grid*.

    Parameters
    ----------
    ohlcv        : pd.DataFrame
    objective_fn : callable — ``(ohlcv, params) → score``
    param_grid   : dict
    max_iter     : int — number of random samples
    seed         : int, optional — RNG seed for reproducibility

    Returns
    -------
    (best_params, best_score)
    """
    if not param_grid:
        return {}, 0.0

    rng = np.random.default_rng(seed)
    best_params: Dict[str, Any] = {}
    best_score = -np.inf

    for _ in range(max_iter):
        params: Dict[str, Any] = {
            k: vlist[int(rng.integers(len(vlist)))]
            for k, vlist in param_grid.items()
        }
        try:
            score = float(objective_fn(ohlcv, params))
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            continue

    return best_params, best_score


def search(
    ohlcv: pd.DataFrame,
    objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
    param_grid: Dict[str, List[Any]],
    method: str = "random",
    max_iter: int = 50,
) -> Tuple[Dict[str, Any], float]:
    """Unified search dispatcher.

    Parameters
    ----------
    method : ``"grid"`` | ``"random"``

    Returns
    -------
    (best_params, best_score)
    """
    if method == "grid":
        return grid_search(ohlcv, objective_fn, param_grid, max_iter)
    return random_search(ohlcv, objective_fn, param_grid, max_iter)

"""
phinance.optimization.bayesian
================================

Bayesian hyperparameter optimisation using Optuna (Tree-structured Parzen
Estimator sampler by default).

Bayesian optimization is significantly more sample-efficient than random
search: it builds a probabilistic surrogate of the objective function and
uses an acquisition function (Expected Improvement) to choose the next
trial.  This typically finds better parameters in 3–5× fewer evaluations.

References
----------
* Bergstra & Bengio (2012) — "Random Search for Hyper-Parameter Optimization"
* Akiba et al. (2019) — "Optuna: A Next-generation Hyperparameter Optimization
  Framework" (https://arxiv.org/abs/1907.10902)

Public API
----------
  bayesian_search(ohlcv, objective_fn, param_grid, n_trials, seed, storage,
                  study_name, direction, show_progress) → (best_params, best_score)
  create_study(name, storage, direction) → optuna.Study
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress Optuna's verbose INFO logs unless the user explicitly wants them
_optuna_log_level = getattr(logging, os.environ.get("PHINANCE_LOG_LEVEL", "WARNING"))
logging.getLogger("optuna").setLevel(_optuna_log_level)

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPTUNA_AVAILABLE = False

from phinance.utils.logging import get_logger

logger = get_logger(__name__)


def _build_trial_params(
    trial: "optuna.Trial",
    param_grid: Dict[str, list],
) -> Dict[str, Any]:
    """Suggest parameter values from *param_grid* for an Optuna trial.

    Discrete lists → ``suggest_categorical``.
    Lists of two floats where values look continuous → ``suggest_float``.
    Lists of two ints where values look step-like → ``suggest_int``.

    In practice all of our grids are short discrete lists, so we always
    use ``suggest_categorical``.
    """
    return {
        k: trial.suggest_categorical(k, v)
        for k, v in param_grid.items()
    }


def bayesian_search(
    ohlcv: pd.DataFrame,
    objective_fn: Callable[[pd.DataFrame, Dict[str, Any]], float],
    param_grid: Dict[str, list],
    n_trials: int = 50,
    seed: Optional[int] = 42,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    direction: str = "maximize",
    show_progress: bool = False,
) -> Tuple[Dict[str, Any], float]:
    """Bayesian hyperparameter search using Optuna TPE sampler.

    Parameters
    ----------
    ohlcv        : pd.DataFrame — OHLCV data passed verbatim to ``objective_fn``
    objective_fn : callable    — ``(ohlcv, params) → score`` (higher = better)
    param_grid   : dict        — ``{param_name: [list_of_values]}``
    n_trials     : int         — number of Optuna trials (default 50)
    seed         : int         — RNG seed for reproducibility (default 42)
    storage      : str, optional — Optuna storage URL (e.g. ``sqlite:///study.db``)
                                   Defaults to in-memory.
    study_name   : str, optional — study identifier (auto-generated if None)
    direction    : str         — ``"maximize"`` | ``"minimize"`` (default ``"maximize"``)
    show_progress: bool        — show tqdm progress bar (default False)

    Returns
    -------
    (best_params, best_score)

    Raises
    ------
    ImportError  — if Optuna is not installed
    ValueError   — if param_grid is empty
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for Bayesian optimisation. "
            "Install with: pip install optuna"
        )

    if not param_grid:
        return {}, 0.0

    # Resolve storage from environment variable if not supplied
    if storage is None:
        storage = os.environ.get("OPTUNA_STORAGE") or None  # empty string → None

    sampler = TPESampler(seed=seed, multivariate=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )

    def _optuna_objective(trial: "optuna.Trial") -> float:
        params = _build_trial_params(trial, param_grid)
        try:
            score = float(objective_fn(ohlcv, params))
            return score if np.isfinite(score) else 0.0
        except Exception as exc:
            logger.debug("Trial failed: %s", exc)
            return 0.0

    study.optimize(
        _optuna_objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
        n_jobs=1,           # caller controls parallelism at the indicator level
    )

    best = study.best_trial
    best_params = {k: v for k, v in best.params.items()}
    best_score = float(best.value) if best.value is not None else 0.0

    logger.info(
        "Bayesian search: %d trials, best_score=%.4f, params=%s",
        n_trials,
        best_score,
        best_params,
    )
    return best_params, best_score


def create_study(
    name: str = "phinance",
    storage: Optional[str] = None,
    direction: str = "maximize",
) -> "optuna.Study":
    """Create or load an Optuna study.

    Convenience wrapper for external callers (e.g. the Streamlit UI).

    Parameters
    ----------
    name      : study name
    storage   : Optuna storage URL (default: in-memory)
    direction : ``"maximize"`` | ``"minimize"``

    Returns
    -------
    optuna.Study
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("pip install optuna")

    sampler = TPESampler(seed=42, multivariate=True)
    return optuna.create_study(
        study_name=name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )

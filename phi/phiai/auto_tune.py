"""
PhiAI Auto-Tuning Engine

Grid/random search over indicator parameters.
Regime-conditioned parameter selection.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

_MAX_PARALLEL_INDICATORS = 4

import numpy as np
import pandas as pd


def auto_tune_params(
    ohlcv: pd.DataFrame,
    indicator_name: str,
    param_grid: Dict[str, List[Any]],
    objective_fn: Callable[[pd.DataFrame, Dict], float],
    max_iter: int = 50,
    method: str = "random",
) -> Tuple[Dict[str, Any], float]:
    """
    Auto-tune indicator parameters via grid or random search.

    Parameters
    ----------
    ohlcv : OHLCV data
    indicator_name : name of indicator
    param_grid : {param_name: [values]}
    objective_fn : (ohlcv, params) -> score (higher is better)
    max_iter : max evaluations
    method : 'grid' | 'random'

    Returns
    -------
    (best_params, best_score)
    """
    best_params = {}
    best_score = -np.inf

    if method == "grid":
        keys = list(param_grid.keys())
        vals = list(param_grid.values())
        from itertools import product

        combos = list(product(*vals))[:max_iter]
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                score = objective_fn(ohlcv, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue
    else:
        for _ in range(max_iter):
            params = {}
            for k, vlist in param_grid.items():
                params[k] = vlist[np.random.randint(len(vlist))]
            try:
                score = objective_fn(ohlcv, params)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

    return best_params, best_score


class PhiAI:
    """PhiAI orchestrator for full auto mode."""

    def __init__(
        self,
        max_indicators: int = 5,
        allow_shorts: bool = False,
        risk_cap: Optional[float] = None,
    ) -> None:
        """Initialise the PhiAI orchestrator.

        Parameters
        ----------
        max_indicators : int
            Maximum number of indicators to enable simultaneously (default 5).
        allow_shorts : bool
            Whether the strategy is permitted to short (default False).
        risk_cap : float, optional
            Maximum allowable portfolio risk as a fraction (e.g. 0.02 = 2%).
        """
        self.max_indicators = max_indicators
        self.allow_shorts = allow_shorts
        self.risk_cap = risk_cap
        self.changes: List[Dict[str, str]] = []

    def explain(self) -> str:
        """Return a detailed explanation of all PhiAI adjustments.

        Returns
        -------
        str
            Human-readable summary of every change recorded in ``self.changes``,
            including *what* was changed, *why*, and configuration constraints
            (``max_indicators``, ``allow_shorts``, ``risk_cap``).

        Examples
        --------
        >>> ai = PhiAI(max_indicators=3)
        >>> ai.changes = [{"what": "RSI params", "reason": "Optimized → acc 62.0%"}]
        >>> print(ai.explain())
        """
        lines = [
            f"PhiAI configuration: max_indicators={self.max_indicators}, "
            f"allow_shorts={self.allow_shorts}, risk_cap={self.risk_cap}",
        ]
        if not self.changes:
            lines.append("No adjustments were made.")
        else:
            lines.append(f"{len(self.changes)} adjustment(s):")
            for c in self.changes:
                what = c.get("what", "unknown")
                reason = c.get("reason", "")
                lines.append(f"  • {what}: {reason}")
        return "\n".join(lines)


def run_phiai_optimization(
    ohlcv: pd.DataFrame,
    indicators: Dict[str, Dict[str, Any]],
    max_iter_per_indicator: int = 20,
    timeframe: str = "1D",
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Auto-tune parameters for every indicator using a direction-accuracy proxy.

    Each indicator's parameter grid is searched in parallel using
    ``concurrent.futures.ThreadPoolExecutor``, then the best-scoring
    parameter set is selected and stored.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        OHLCV data used as the optimisation dataset.  Must contain a
        ``close`` column.
    indicators : dict[str, dict]
        Mapping of indicator name → config dict
        (``{"enabled": bool, "auto_tune": bool, "params": {...}}``).
    max_iter_per_indicator : int
        Maximum number of random-search evaluations per indicator.

    Returns
    -------
    tuple[dict[str, dict], str]
        ``(optimized_indicators, explanation)`` where ``optimized_indicators``
        mirrors *indicators* with updated params, and ``explanation`` is a
        human-readable summary of all changes made.

    Examples
    --------
    >>> ohlcv = pd.DataFrame({"close": [100, 101, 99, 102]})
    >>> cfg = {"RSI": {"enabled": True, "auto_tune": True, "params": {}}}
    >>> opt, msg = run_phiai_optimization(ohlcv, cfg, max_iter_per_indicator=5)
    """
    from phi.indicators.simple import compute_indicator, INDICATOR_COMPUTERS

    optimized = {}
    changes = []

    # Intraday timeframes need shorter periods to generate signals within 10m–10h.
    _INTRADAY_TF = {"1m", "5m", "15m", "30m", "1H"}
    if timeframe in _INTRADAY_TF:
        param_grids = {
            "RSI": {"rsi_period": [3, 5, 7, 9, 14], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
            "MACD": {"fast_period": [3, 5, 8], "slow_period": [12, 17, 21], "signal_period": [3, 5, 7]},
            "Bollinger": {"bb_period": [10, 14, 20], "num_std": [1.5, 2.0, 2.5]},
            "Dual SMA": {"fast_period": [3, 5, 9], "slow_period": [12, 20, 30]},
            "Mean Reversion": {"sma_period": [5, 10, 15]},
            "Breakout": {"channel_period": [5, 10, 15]},
            "VWAP": {"band_pct": [0.2, 0.3, 0.5, 0.8, 1.0]},
        }
    else:
        param_grids = {
            "RSI": {"rsi_period": [7, 14, 21], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
            "MACD": {"fast_period": [8, 12, 16], "slow_period": [21, 26, 31], "signal_period": [7, 9, 11]},
            "Bollinger": {"bb_period": [15, 20, 25], "num_std": [1.5, 2.0, 2.5]},
            "Dual SMA": {"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
            "Mean Reversion": {"sma_period": [10, 20, 40]},
            "Breakout": {"channel_period": [10, 20, 40]},
            "VWAP": {"band_pct": [0.2, 0.5, 1.0]},
        }

    close = ohlcv["close"].values
    direction = np.zeros(len(close) - 1)
    direction[close[1:] > close[:-1]] = 1
    direction[close[1:] < close[:-1]] = -1

    def _make_objective(ind_name: str):
        def _obj(ohlcv_df: pd.DataFrame, params: Dict) -> float:
            try:
                sig = compute_indicator(ind_name, ohlcv_df, params)
                if sig is None or len(sig) < 10:
                    return 0.0
                sig_aligned = sig.iloc[:-1].values
                if len(sig_aligned) != len(direction):
                    n = min(len(sig_aligned), len(direction))
                    sig_aligned = sig_aligned[-n:]
                    dir_use = direction[-n:]
                else:
                    dir_use = direction
                dir_pred = np.sign(sig_aligned)
                dir_pred[dir_pred == 0] = 1
                matches = np.sum((dir_pred * dir_use) > 0)
                return matches / max(1, len(dir_use))
            except Exception:
                return 0.0
        return _obj

    import concurrent.futures

    def _tune_one(item):
        name, cfg = item
        if name not in param_grids or name not in INDICATOR_COMPUTERS:
            return name, cfg, None
        grid = param_grids[name]
        best_params, best_score = auto_tune_params(
            ohlcv, name, grid,
            _make_objective(name),
            max_iter=max_iter_per_indicator,
            method="random",
        )
        return name, cfg, (best_params, best_score)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(indicators), _MAX_PARALLEL_INDICATORS) or 1) as executor:
        futures = list(executor.map(_tune_one, indicators.items()))

    for name, cfg, result in futures:
        if result is None:
            optimized[name] = cfg
        else:
            best_params, best_score = result
            if best_params:
                optimized[name] = {"enabled": True, "auto_tune": False, "params": best_params}
                changes.append({"what": f"{name} params", "reason": f"Optimized → acc {best_score:.1%}"})
            else:
                optimized[name] = cfg

    explanation = "PhiAI adjustments:\n" + "\n".join(f"- {c['what']}: {c['reason']}" for c in changes) if changes else "PhiAI made no changes."
    return optimized, explanation

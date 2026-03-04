"""
phi.phiai.tuner — PhiAI Auto-Tuner
====================================
Optimizes indicator parameters and indicator selection using random search.

Strategy:
  1. Define parameter ranges for each indicator
  2. Random sample N configurations
  3. Evaluate each using chosen metric (via fast vectorized backtest)
  4. Return best params + explanation
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Regime classifier (minimal — volatility + trend)
# ─────────────────────────────────────────────────────────────────────────────

def classify_simple_regime(ohlcv: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Minimal 3-state regime classifier.
    Returns Series: 'TREND_UP' | 'TREND_DN' | 'RANGE'
    """
    close = ohlcv["close"].astype(float)
    ma    = close.rolling(window, min_periods=window // 2).mean()
    std   = close.rolling(window, min_periods=window // 2).std().clip(lower=1e-10)

    atr   = (ohlcv["high"] - ohlcv["low"]).astype(float).rolling(window, min_periods=5).mean()
    vol_z = _safe_zscore(atr, window * 3)

    regime = pd.Series("RANGE", index=ohlcv.index)
    regime[close > ma + 0.5 * std] = "TREND_UP"
    regime[close < ma - 0.5 * std] = "TREND_DN"

    return regime


def _safe_zscore(s: pd.Series, w: int) -> pd.Series:
    mu  = s.rolling(w, min_periods=w // 3).mean()
    std = s.rolling(w, min_periods=w // 3).std().clip(lower=1e-10)
    return (s - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_params(tune_ranges: Dict[str, Tuple], current: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a random parameter set from tune_ranges."""
    params = deepcopy(current)
    for key, (lo, hi) in tune_ranges.items():
        if isinstance(lo, int) and isinstance(hi, int):
            params[key] = random.randint(lo, hi)
        else:
            params[key] = random.uniform(lo, hi)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Metric extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    """Extract a scalar metric value (higher = better)."""
    v = metrics.get(metric_name, None)
    if v is None:
        return -999.0
    v = float(v)

    # For max_drawdown: less negative is better — negate
    if metric_name == "max_drawdown":
        return -abs(v)

    # For win_rate: already 0–1
    return v


# ─────────────────────────────────────────────────────────────────────────────
# PhiAI Tuner
# ─────────────────────────────────────────────────────────────────────────────

class PhiAITuner:
    """
    Random-search parameter tuner with optional drawdown constraint.

    Parameters
    ----------
    n_trials          : number of random configurations to try
    metric            : metric to maximize ('sharpe', 'roi', 'cagr', etc.)
    max_drawdown_cap  : reject configs with drawdown worse than this (e.g. -0.25)
    no_short          : disallow shorting
    max_indicators    : max number of indicators to keep enabled
    random_seed       : for reproducibility
    """

    def __init__(
        self,
        n_trials:         int   = 50,
        metric:           str   = "sharpe",
        max_drawdown_cap: float = -0.50,
        no_short:         bool  = True,
        max_indicators:   int   = 5,
        random_seed:      Optional[int] = None,
    ) -> None:
        self.n_trials         = n_trials
        self.metric           = metric
        self.max_drawdown_cap = max_drawdown_cap
        self.no_short         = no_short
        self.max_indicators   = max_indicators

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def tune_indicator(
        self,
        indicator_name: str,
        ohlcv: pd.DataFrame,
        current_params: Dict[str, Any],
        tune_ranges: Dict[str, Tuple],
        backtest_engine_cfg: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[Dict[str, Any], float, str]:
        """
        Tune parameters for a single indicator.

        Returns
        -------
        (best_params, best_score, explanation)
        """
        from phi.indicators.registry import compute_signal
        from phi.backtest.engine import BacktestEngine

        engine = BacktestEngine(backtest_engine_cfg)
        best_params = deepcopy(current_params)
        best_score  = -999.0
        results_log: List[Tuple[Dict, float]] = []

        for trial in range(self.n_trials):
            params = _sample_params(tune_ranges, current_params)

            try:
                sig  = compute_signal(indicator_name, ohlcv, params)
                res  = engine.run(ohlcv, sig)
                m    = res.metrics

                # Drawdown constraint
                dd = m.get("max_drawdown", -1.0)
                if dd < self.max_drawdown_cap:
                    continue

                score = _extract_metric(m, self.metric)
                results_log.append((deepcopy(params), score))

                if score > best_score:
                    best_score  = score
                    best_params = deepcopy(params)

            except Exception:
                pass

            if progress_callback:
                progress_callback(trial + 1, self.n_trials)

        # Build explanation
        explanation = self._explain_tune(
            indicator_name, best_params, best_score, self.metric, len(results_log)
        )
        return best_params, best_score, explanation

    def select_indicators(
        self,
        indicator_names: List[str],
        ohlcv: pd.DataFrame,
        params_map: Dict[str, Dict[str, Any]],
        backtest_engine_cfg: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[str], Dict[str, float], str]:
        """
        Auto-select best subset of indicators.

        Returns
        -------
        (selected_names, metric_scores_per_indicator, explanation)
        """
        from phi.indicators.registry import compute_signal
        from phi.backtest.engine import BacktestEngine

        engine = BacktestEngine(backtest_engine_cfg)
        ind_scores: Dict[str, float] = {}

        # Score each indicator individually
        for idx, name in enumerate(indicator_names):
            try:
                sig = compute_signal(name, ohlcv, params_map.get(name, {}))
                res = engine.run(ohlcv, sig)
                dd  = res.metrics.get("max_drawdown", -1.0)
                if dd >= self.max_drawdown_cap:
                    ind_scores[name] = _extract_metric(res.metrics, self.metric)
                else:
                    ind_scores[name] = -999.0
            except Exception:
                ind_scores[name] = -999.0

            if progress_callback:
                progress_callback(idx + 1, len(indicator_names))

        # Select top-N
        sorted_inds = sorted(ind_scores.items(), key=lambda x: x[1], reverse=True)
        selected    = [name for name, score in sorted_inds
                       if score > -999.0][:self.max_indicators]

        if not selected:
            selected = indicator_names[:self.max_indicators]

        explanation = self._explain_selection(
            sorted_inds, selected, self.metric
        )
        return selected, ind_scores, explanation

    # ─────────────────────────────────────────────────────────────────────
    # Explanation builders
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _explain_tune(
        name: str,
        best_params: Dict[str, Any],
        best_score: float,
        metric: str,
        n_tested: int,
    ) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
        return (
            f"PhiAI tested {n_tested} configurations for {name}. "
            f"Best {metric} = {best_score:.4f} with params: [{param_str}]. "
            f"Parameters were selected to maximize {metric} while respecting drawdown constraints."
        )

    @staticmethod
    def _explain_selection(
        scored: List[Tuple[str, float]],
        selected: List[str],
        metric: str,
    ) -> str:
        lines = [f"PhiAI evaluated {len(scored)} indicators on {metric}:"]
        for name, score in scored[:8]:
            mark = "✓" if name in selected else "✗"
            score_str = f"{score:.4f}" if score > -999 else "rejected (drawdown)"
            lines.append(f"  {mark} {name}: {score_str}")
        lines.append(f"Selected: {', '.join(selected)}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function for the workbench
# ─────────────────────────────────────────────────────────────────────────────

def phiai_full_auto(
    ohlcv: pd.DataFrame,
    indicator_names: List[str],
    params_map: Dict[str, Dict[str, Any]],
    tune_ranges_map: Dict[str, Dict[str, Tuple]],
    backtest_cfg: Dict[str, Any],
    constraints: Dict[str, Any],
    metric: str = "sharpe",
    n_trials: int = 40,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """
    Run full PhiAI: select indicators + tune parameters.

    Returns dict with:
      selected_indicators : list of indicator names
      best_params         : {indicator_name: {param: value}}
      metric_scores       : {indicator_name: score}
      blend_weights       : {indicator_name: weight}
      explanation         : str
    """
    def _prog(step, pct):
        if progress_callback:
            progress_callback(step, pct)

    tuner = PhiAITuner(
        n_trials         = n_trials,
        metric           = metric,
        max_drawdown_cap = float(constraints.get("max_drawdown_cap", -0.50)),
        no_short         = bool(constraints.get("no_short", True)),
        max_indicators   = int(constraints.get("max_indicators", 3)),
    )

    _prog("PhiAI: Scoring indicators...", 0.1)

    # Step 1: Select best indicators
    selected, scores, sel_expl = tuner.select_indicators(
        indicator_names, ohlcv, params_map, backtest_cfg
    )
    _prog("PhiAI: Tuning parameters...", 0.4)

    # Step 2: Tune params for each selected indicator
    best_params: Dict[str, Dict[str, Any]] = {}
    tune_explanations: List[str] = []

    for idx, name in enumerate(selected):
        ranges = tune_ranges_map.get(name, {})
        current = deepcopy(params_map.get(name, {}))

        if ranges:
            bp, score, expl = tuner.tune_indicator(
                name, ohlcv, current, ranges, backtest_cfg
            )
            best_params[name] = bp
            tune_explanations.append(expl)
        else:
            best_params[name] = current

        _prog(f"PhiAI: Tuned {name}...", 0.4 + 0.5 * (idx + 1) / max(len(selected), 1))

    # Step 3: Compute blend weights from scores
    valid_scores = {k: max(v, 0.0) for k, v in scores.items() if k in selected and v > -999}
    total = sum(valid_scores.values()) + 1e-10
    blend_weights = {k: v / total for k, v in valid_scores.items()}

    explanation = sel_expl + "\n\n" + "\n".join(tune_explanations)
    _prog("PhiAI: Done", 1.0)

    return {
        "selected_indicators": selected,
        "best_params":         best_params,
        "metric_scores":       scores,
        "blend_weights":       blend_weights,
        "explanation":         explanation,
    }

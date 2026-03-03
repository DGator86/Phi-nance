"""
phinance.optimization.walk_forward
====================================

Walk-Forward Optimisation (WFO) — rolling in-sample/out-of-sample testing.

Walk-forward optimisation splits the full history into successive windows:
  1. **In-sample** (training) window — optimise indicator parameters.
  2. **Out-of-sample** (test)     window — run backtest with the best params.
  3. Roll the window forward by ``step_bars`` bars and repeat.

This avoids look-ahead bias in parameter tuning and gives a realistic picture
of how a strategy would have performed in live trading.

Usage
-----
    from phinance.optimization.walk_forward import walk_forward_optimize

    results = walk_forward_optimize(
        ohlcv       = df,
        symbol      = "SPY",
        indicators  = {"RSI": {"enabled": True, "params": {}}},
        in_window   = 252,   # trading days (~1 year)
        out_window  = 63,    # ~1 quarter
        step_bars   = 63,
        n_trials    = 40,
        metric      = "sharpe",
    )
    print(results.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from phinance.utils.logging import get_logger

_log = get_logger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class WFOWindow:
    """Result for a single walk-forward window."""
    window_idx:     int
    in_start:       str
    in_end:         str
    out_start:      str
    out_end:        str
    best_params:    Dict[str, Any]
    in_sample_score: float
    out_of_sample_metrics: Dict[str, float]


@dataclass
class WFOResult:
    """Aggregated walk-forward result."""
    windows:       List[WFOWindow] = field(default_factory=list)
    symbol:        str = ""
    metric:        str = "sharpe"

    # ── Aggregate stats ───────────────────────────────────────────────────────

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    @property
    def oos_returns(self) -> List[float]:
        return [w.out_of_sample_metrics.get("total_return", 0.0) for w in self.windows]

    @property
    def oos_sharpes(self) -> List[float]:
        return [w.out_of_sample_metrics.get("sharpe", 0.0) for w in self.windows]

    @property
    def mean_oos_return(self) -> float:
        r = self.oos_returns
        return float(np.mean(r)) if r else 0.0

    @property
    def mean_oos_sharpe(self) -> float:
        s = self.oos_sharpes
        return float(np.mean(s)) if s else 0.0

    @property
    def consistency_ratio(self) -> float:
        """Fraction of OOS windows that were profitable."""
        r = self.oos_returns
        if not r:
            return 0.0
        return sum(1 for x in r if x > 0) / len(r)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "symbol":              self.symbol,
            "metric":              self.metric,
            "n_windows":           self.n_windows,
            "mean_oos_return":     round(self.mean_oos_return, 4),
            "mean_oos_sharpe":     round(self.mean_oos_sharpe, 4),
            "consistency_ratio":   round(self.consistency_ratio, 4),
            "oos_returns":         [round(r, 4) for r in self.oos_returns],
            "oos_sharpes":         [round(s, 4) for s in self.oos_sharpes],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return OOS metrics as a tidy DataFrame."""
        rows = []
        for w in self.windows:
            row = {
                "window":     w.window_idx,
                "out_start":  w.out_start,
                "out_end":    w.out_end,
                "in_score":   w.in_sample_score,
                **{f"oos_{k}": v for k, v in w.out_of_sample_metrics.items()},
            }
            rows.append(row)
        return pd.DataFrame(rows)


# ── Core engine ───────────────────────────────────────────────────────────────

def walk_forward_optimize(
    ohlcv:      pd.DataFrame,
    symbol:     str = "ASSET",
    indicators: Optional[Dict[str, Any]] = None,
    blend_method: str = "weighted_sum",
    in_window:  int = 252,
    out_window: int = 63,
    step_bars:  int = 63,
    n_trials:   int = 30,
    metric:     str = "sharpe",
    initial_capital: float = 100_000.0,
) -> WFOResult:
    """Run walk-forward optimisation over a full history.

    Parameters
    ----------
    ohlcv         : pd.DataFrame — OHLCV data (normalised columns).
    symbol        : str          — ticker symbol (for logging).
    indicators    : dict         — ``{name: {"enabled": bool, "params": {}}}``
    blend_method  : str          — blending method.
    in_window     : int          — in-sample bars per fold.
    out_window    : int          — out-of-sample bars per fold.
    step_bars     : int          — roll-forward increment (bars).
    n_trials      : int          — random-search trials per fold.
    metric        : str          — optimisation target metric.
    initial_capital : float

    Returns
    -------
    WFOResult
    """
    from phinance.optimization.grid_search import random_search
    from phinance.backtest.runner import run_backtest
    from phinance.strategies.params import DAILY_GRIDS

    indicators = indicators or {"RSI": {"enabled": True, "params": {}}}
    result = WFOResult(symbol=symbol, metric=metric)

    total_bars = len(ohlcv)
    start_idx  = 0
    window_idx = 0

    _log.info(
        "WFO start: symbol=%s, in=%d, out=%d, step=%d, total=%d bars",
        symbol, in_window, out_window, step_bars, total_bars,
    )

    while start_idx + in_window + out_window <= total_bars:
        in_slice  = ohlcv.iloc[start_idx : start_idx + in_window]
        out_slice = ohlcv.iloc[start_idx + in_window : start_idx + in_window + out_window]

        # ── In-sample: random-search for best params ──────────────────────────
        best_params: Dict[str, Any] = {}
        best_score: float = -np.inf

        enabled_names = [
            n for n, cfg in indicators.items()
            if cfg.get("enabled", True)
        ]

        for _ in range(n_trials):
            trial_params: Dict[str, Any] = {}
            for ind_name in enabled_names:
                grid = DAILY_GRIDS.get(ind_name, {})
                trial_params[ind_name] = {
                    k: (np.random.choice(v) if isinstance(v, list) else v)
                    for k, v in grid.items()
                }

            ind_cfg: Dict[str, Any] = {
                n: {"enabled": True, "params": trial_params.get(n, {})}
                for n in enabled_names
            }

            try:
                br = run_backtest(
                    ohlcv=in_slice,
                    symbol=symbol,
                    indicators=ind_cfg,
                    blend_method=blend_method,
                    initial_capital=initial_capital,
                )
                score = _get_metric(br, metric)
                if score > best_score:
                    best_score = score
                    best_params = trial_params
            except Exception:
                continue

        # ── Out-of-sample: apply best params ──────────────────────────────────
        oos_ind_cfg = {
            n: {"enabled": True, "params": best_params.get(n, {})}
            for n in enabled_names
        }
        try:
            oos_br = run_backtest(
                ohlcv=out_slice,
                symbol=symbol,
                indicators=oos_ind_cfg,
                blend_method=blend_method,
                initial_capital=initial_capital,
            )
            oos_metrics = _extract_oos_metrics(oos_br)
        except Exception as exc:
            _log.warning("WFO window %d OOS failed: %s", window_idx, exc)
            oos_metrics = {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        in_dt  = ohlcv.index
        window = WFOWindow(
            window_idx=window_idx,
            in_start=str(in_dt[start_idx])[:10],
            in_end=str(in_dt[start_idx + in_window - 1])[:10],
            out_start=str(in_dt[start_idx + in_window])[:10],
            out_end=str(in_dt[min(start_idx + in_window + out_window - 1, len(in_dt) - 1)])[:10],
            best_params=best_params,
            in_sample_score=round(best_score, 4),
            out_of_sample_metrics=oos_metrics,
        )
        result.windows.append(window)
        _log.info(
            "WFO window %d: in_score=%.4f, oos_return=%.4f, oos_sharpe=%.4f",
            window_idx, best_score,
            oos_metrics.get("total_return", 0.0),
            oos_metrics.get("sharpe", 0.0),
        )

        start_idx  += step_bars
        window_idx += 1

    _log.info(
        "WFO complete: %d windows, mean_oos_return=%.4f, consistency=%.2f",
        result.n_windows, result.mean_oos_return, result.consistency_ratio,
    )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_metric(backtest_result: Any, metric: str) -> float:
    """Extract a scalar metric from a BacktestResult."""
    d = backtest_result.to_dict() if hasattr(backtest_result, "to_dict") else {}
    mapping = {
        "sharpe":       d.get("sharpe", 0.0),
        "sortino":      d.get("sortino", 0.0),
        "roi":          d.get("total_return", 0.0),
        "total_return": d.get("total_return", 0.0),
        "cagr":         d.get("cagr", 0.0),
        "win_rate":     d.get("win_rate", 0.0),
    }
    val = mapping.get(metric, d.get(metric, 0.0))
    return float(val) if val is not None else 0.0


def _extract_oos_metrics(backtest_result: Any) -> Dict[str, float]:
    """Convert BacktestResult to a flat dict of OOS metrics."""
    d = backtest_result.to_dict() if hasattr(backtest_result, "to_dict") else {}
    return {
        "total_return": round(float(d.get("total_return", 0.0)), 4),
        "cagr":         round(float(d.get("cagr", 0.0)), 4),
        "sharpe":       round(float(d.get("sharpe", 0.0)), 4),
        "sortino":      round(float(d.get("sortino", 0.0)), 4),
        "max_drawdown": round(float(d.get("max_drawdown", 0.0)), 4),
        "win_rate":     round(float(d.get("win_rate", 0.0)), 4),
        "total_trades": int(d.get("total_trades", 0)),
    }

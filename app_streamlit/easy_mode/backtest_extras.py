"""Heavy optional easy-mode analysis: multi-window regimes, bootstrap robustness, RSI x threshold heatmap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app_streamlit.config import INDICATOR_SPECS, IndicatorSpec
from app_streamlit.easy_mode.backtest_core import (
    call_run_direct_backtest,
    default_params_from_spec,
    semantic_label_map_for_clusters,
)
from phi.indicators.simple import INDICATOR_COMPUTERS
from phi.regime.train import train_regime_detector

PROBE_NAMES = ("RSI", "MACD", "Bollinger", "Dual SMA", "Buy & Hold")


def train_multi_window_regimes(
    ohlcv: pd.DataFrame,
    *,
    windows: tuple[tuple[int, str], ...] = ((12, "SHORT"), (20, "MEDIUM"), (40, "LONG")),
) -> dict[str, dict[str, Any]]:
    """Three k-means detectors with different feature windows (NPR-style tiers)."""
    out: dict[str, dict[str, Any]] = {}
    for w, tag in windows:
        det, _ = train_regime_detector(ohlcv, method="kmeans", n_regimes=3, window=int(w), save=False)
        rs = det.predict(ohlcv)
        rmap = semantic_label_map_for_clusters(ohlcv, rs)
        out[tag] = {"window": w, "series": rs, "label_map": rmap}
    return out


def _sharpe_from_pv(pv: list[float], bars_per_year: float) -> float:
    arr = np.asarray(pv, dtype=float)
    if len(arr) < 3:
        return 0.0
    r = np.diff(arr) / (arr[:-1] + 1e-12)
    if np.std(r) <= 0:
        return 0.0
    return float(np.mean(r) / np.std(r) * np.sqrt(bars_per_year))


def bootstrap_sharpe_distribution(
    portfolio_value: list[float],
    *,
    n_sims: int = 350,
    bars_per_year: float = 252.0,
    seed: int = 42,
) -> dict[str, float]:
    """Shuffle bar-to-bar returns to approximate sampling distribution of Sharpe (sanity / robustness)."""
    pv = list(portfolio_value)
    if len(pv) < 10:
        return {"original": 0.0, "mean": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    rng = np.random.default_rng(seed)
    r = np.diff(np.asarray(pv, dtype=float)) / (np.asarray(pv[:-1], dtype=float) + 1e-12)
    orig = _sharpe_from_pv(pv, bars_per_year)
    sims: list[float] = []
    for _ in range(n_sims):
        perm = rng.permutation(r)
        eq = [float(pv[0])]
        for x in perm:
            eq.append(eq[-1] * (1.0 + float(x)))
        sims.append(_sharpe_from_pv(eq, bars_per_year))
    a = np.asarray(sims, dtype=float)
    return {
        "original": orig,
        "mean": float(np.mean(a)),
        "p05": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def _build_probe_indicators(rsi_period: int) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    names = [n for n in PROBE_NAMES if n in INDICATOR_SPECS and n in INDICATOR_COMPUTERS]
    indicators: dict[str, dict[str, Any]] = {}
    for name in names:
        spec: IndicatorSpec = INDICATOR_SPECS[name]
        params = default_params_from_spec(spec)
        if name == "RSI":
            params["rsi_period"] = int(rsi_period)
        indicators[name] = {"enabled": True, "params": params}
    n = len(indicators)
    w = round(1.0 / n, 6) if n else 1.0
    weights = {k: w for k in indicators}
    if weights:
        first = next(iter(weights))
        weights[first] = round(w + (1.0 - sum(weights.values())), 6)
    return indicators, weights


def run_rsi_threshold_heatmap(
    ohlcv: pd.DataFrame,
    regime_series: pd.Series,
    regime_label_map: dict[str, str],
    regime_boosts: dict[str, dict[str, float]],
    *,
    rsi_periods: tuple[int, ...] = (10, 12, 14, 18, 22),
    thresholds: tuple[float, ...] = (0.10, 0.14, 0.18, 0.22, 0.26),
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Grid Sharpe (probe stack). Rows = threshold labels, Cols = RSI period labels."""
    rows: list[list[float]] = []
    for th in thresholds:
        row: list[float] = []
        for rp in rsi_periods:
            inds, wts = _build_probe_indicators(rp)
            if not inds:
                row.append(float("nan"))
                continue
            res, _ = call_run_direct_backtest(
                ohlcv=ohlcv,
                symbol=symbol,
                indicators=inds,
                blend_weights=dict(wts),
                blend_method="regime_weighted",
                signal_threshold=float(th),
                initial_capital=100_000.0,
                position_size_pct=0.95,
                regime_series=regime_series,
                regime_label_map=regime_label_map or None,
                regime_boosts=regime_boosts,
            )
            row.append(float(res.get("sharpe", 0.0) or 0.0))
        rows.append(row)
    return pd.DataFrame(rows, index=[str(t) for t in thresholds], columns=[str(p) for p in rsi_periods])


def slice_replay_window(
    ohlcv: pd.DataFrame,
    mapped_regimes: pd.Series,
    best: dict[str, Any],
    end_ix: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Truncate OHLCV / regimes / equity / trades for playback scrubber."""
    end_ix = max(1, min(end_ix, len(ohlcv)))
    sub = ohlcv.iloc[:end_ix]
    m = mapped_regimes.reindex(sub.index).ffill()
    pv = best.get("portfolio_value") or []
    if len(pv) >= end_ix + 1:
        pv_sub = list(pv[: end_ix + 1])
    else:
        pv_sub = list(pv)
    last_ts = sub.index[-1]
    te = [
        t
        for t in best.get("trade_events") or []
        if pd.Timestamp(t["date"]) <= pd.Timestamp(last_ts)
    ]
    out = {**best, "portfolio_value": pv_sub, "trade_events": te}
    return sub, m, out

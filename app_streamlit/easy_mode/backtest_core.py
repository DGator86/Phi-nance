"""Shared easy-mode backtest helpers (imported by backtest_screen and backtest_extras)."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd

from app_streamlit.config import INDICATOR_SPECS, IndicatorSpec
from phi.backtest.direct import run_direct_backtest
from phi.indicators.simple import INDICATOR_COMPUTERS

PRESETS: tuple[tuple[str, float], ...] = (
    ("Cautious (fewer trades)", 0.22),
    ("Balanced", 0.16),
    ("Responsive (more trades)", 0.10),
)


def default_params_from_spec(spec: IndicatorSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for param, ps in spec.params.items():
        if isinstance(ps, tuple) and len(ps) >= 3:
            out[param] = ps[2]
        elif isinstance(ps, dict) and ps.get("type") == "select":
            opts = list(ps.get("options", []))
            dv = ps.get("default", opts[0] if opts else "")
            out[param] = dv
    return out


def build_auto_indicators_and_weights() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    names = sorted(set(INDICATOR_SPECS.keys()) & set(INDICATOR_COMPUTERS.keys()))
    indicators = {
        n: {"enabled": True, "params": default_params_from_spec(INDICATOR_SPECS[n])} for n in names
    }
    if not indicators:
        raise RuntimeError("No indicators available (catalog intersection empty).")
    n = len(indicators)
    w = round(1.0 / n, 6)
    weights = dict.fromkeys(indicators, w)
    first = next(iter(weights))
    weights[first] = round(w + (1.0 - sum(weights.values())), 6)
    return indicators, weights


def semantic_label_map_for_clusters(ohlcv: pd.DataFrame, regime_series: pd.Series) -> dict[str, str]:
    rs = regime_series.reindex(ohlcv.index).dropna().astype(str)
    if rs.empty:
        return {}
    close = ohlcv["close"].astype(float)
    rets = close.pct_change()
    labels = sorted(
        rs.unique(),
        key=lambda s: int(s.rsplit("_", 1)[-1]) if str(s).startswith("cluster_") else str(s),
    )
    scored: list[tuple[str, float]] = []
    for lab in labels:
        m = rs == lab
        seg = rets.where(m).dropna()
        scored.append((lab, float(seg.mean()) if len(seg) else 0.0))
    scored.sort(key=lambda x: x[1])
    templates = ("TREND_DN", "RANGE", "TREND_UP")
    n = len(scored)
    if n == 1:
        return {scored[0][0]: "RANGE"}
    if n == 2:
        return {scored[0][0]: "TREND_DN", scored[1][0]: "TREND_UP"}
    out: dict[str, str] = {}
    for i, (lab, _) in enumerate(scored):
        if n == 3:
            out[lab] = templates[i]
        else:
            bucket = min(2, int(3 * i / max(n - 1, 1)))
            out[lab] = templates[bucket]
    return out


def call_run_direct_backtest(**kwargs: Any) -> tuple[dict[str, Any], Any]:
    sig = inspect.signature(run_direct_backtest)
    allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if kwargs.get("blend_method") == "regime_weighted" and "regime_series" not in allowed:
        filtered["blend_method"] = "weighted_sum"
        for drop in (
            "regime_series",
            "regime_label_map",
            "regime_boosts",
            "regime_detector",
            "regime_detector_params",
        ):
            filtered.pop(drop, None)
    return run_direct_backtest(**filtered)



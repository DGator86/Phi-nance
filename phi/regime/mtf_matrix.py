"""Multi-timeframe regime matrix: same k-means/semantic pipeline on resampled OHLCV.

Easy mode's ``train_multi_window_regimes`` varies the *feature window* on one bar series.
This module varies the *bar interval* (1D → 1W, etc.) when the data supports it.

With **daily** data you get e.g. 1D / 1W / 1ME — not 1m/5m unless the input index is intraday.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from phi.regime.train import train_regime_detector

DEFAULT_TIMEFRAME_RULES: tuple[str, ...] = (
    "1min",
    "5min",
    "15min",
    "30min",
    "1H",
    "4H",
    "1D",
    "1W",
    "1ME",
)


def _semantic_label_map_for_clusters(ohlcv: pd.DataFrame, regime_series: pd.Series) -> dict[str, str]:
    """Map cluster ids to TREND_DN / RANGE / TREND_UP by mean forward return (same logic as easy_mode/backtest_core)."""
    rs = regime_series.reindex(ohlcv.index).dropna().astype(str)
    if rs.empty:
        return {}
    close = ohlcv["close"].astype(float)
    rets = close.pct_change()
    labels = sorted(
        rs.unique(),
        key=lambda s: int(str(s).rsplit("_", 1)[-1]) if str(s).startswith("cluster_") else str(s),
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


def _median_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(index) < 2:
        return None
    d = pd.Series(index).diff().dropna()
    if d.empty:
        return None
    return d.median()


def _offset_min_delta(rule: str) -> pd.Timedelta:
    """Smallest step for comparing to observed bar spacing (works for non-fixed offsets like Week)."""
    off = pd.tseries.frequencies.to_offset(rule)
    if hasattr(off, "delta"):
        try:
            d = off.delta
            if d is not None:
                return pd.Timedelta(d)
        except (ValueError, TypeError):
            pass
    anchor = pd.Timestamp("2000-01-03 12:00:00")
    return pd.Timedelta(anchor + off - anchor)


def filter_compatible_timeframe_rules(
    index: pd.DatetimeIndex,
    rules: Sequence[str],
    *,
    slack: float = 0.85,
) -> tuple[list[str], dict[str, str]]:
    """Keep rules whose native period is not finer than the observed bar spacing.

    Returns (kept_rules, skip_reasons for dropped rules).
    """
    skip: dict[str, str] = {}
    med = _median_bar_delta(index)
    if med is None or med.value <= 0:
        return [], {"*": "need at least 2 datetime bars with positive spacing"}

    base_ns = float(med.value)
    kept: list[str] = []
    for rule in rules:
        try:
            rd = _offset_min_delta(rule)
        except (ValueError, TypeError) as exc:
            skip[rule] = f"invalid rule: {exc}"
            continue
        rule_ns = float(rd.value)
        # Aggregate-only: regime bar must be >= base bar (allow small slack for irregular sessions)
        if rule_ns + 1 < base_ns * slack:
            skip[rule] = f"finer than data bars (~{med})"
            continue
        kept.append(rule)
    return kept, skip


def resample_ohlcv(ohlcv: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Standard OHLCV aggregation to a pandas offset string."""
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("ohlcv index must be DatetimeIndex")
    df = ohlcv.sort_index()
    pairs = (
        ("open", "first"),
        ("high", "max"),
        ("low", "min"),
        ("close", "last"),
        ("volume", "sum"),
    )
    agg = {c: fn for c, fn in pairs if c in df.columns}
    need = {"open", "high", "low", "close"}
    if not need.issubset(agg.keys()):
        raise ValueError("ohlcv must include open, high, low, close (volume optional)")
    out = df[list(agg.keys())].resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(how="any")


def build_regime_matrix(
    ohlcv: pd.DataFrame,
    *,
    timeframe_rules: Sequence[str] | None = None,
    n_regimes: int = 3,
    feature_window: int = 20,
    method: str = "kmeans",
    min_resampled_bars: int = 40,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one detector per compatible timeframe; align semantic regimes to ``ohlcv`` index.

    Returns
    -------
    matrix
        Columns = timeframe rule strings, values = TREND_DN | RANGE | TREND_UP (ffill from coarser bars).
    meta
        ``skipped``, ``per_timeframe`` diagnostics.
    """
    rules = tuple(timeframe_rules) if timeframe_rules is not None else DEFAULT_TIMEFRAME_RULES
    idx = ohlcv.index
    if not isinstance(idx, pd.DatetimeIndex):
        return pd.DataFrame(index=idx), {"error": "non-datetime index", "skipped": dict.fromkeys(rules, "non-datetime index")}

    compatible, skip_global = filter_compatible_timeframe_rules(idx, rules)
    per_tf: dict[str, Any] = {}
    skipped = dict(skip_global)
    columns: dict[str, pd.Series] = {}

    for rule in compatible:
        try:
            rdf = resample_ohlcv(ohlcv, rule)
        except Exception as exc:  # noqa: BLE001
            skipped[rule] = f"resample failed: {exc}"
            continue
        if len(rdf) < min_resampled_bars:
            skipped[rule] = f"only {len(rdf)} bars after resample (need {min_resampled_bars})"
            continue
        try:
            det, _ = train_regime_detector(
                rdf,
                method=method,
                n_regimes=n_regimes,
                window=int(feature_window),
                save=False,
            )
            raw = det.predict(rdf)
        except Exception as exc:  # noqa: BLE001
            skipped[rule] = f"detector failed: {exc}"
            continue
        cmap = _semantic_label_map_for_clusters(rdf, raw)
        mapped = raw.astype(str).map(lambda x: cmap.get(x, x))
        # align to base index: as-of each base timestamp, use last closed bar on this TF
        aligned = mapped.reindex(idx, method="ffill")
        columns[rule] = aligned
        per_tf[rule] = {"bars": len(rdf), "label_map": cmap}

    if not columns:
        return pd.DataFrame(index=idx), {"skipped": skipped, "per_timeframe": per_tf}

    matrix = pd.DataFrame(columns, index=idx)
    return matrix, {"skipped": skipped, "per_timeframe": per_tf}


def regime_matrix_to_numeric(matrix: pd.DataFrame) -> pd.DataFrame:
    """Encode TREND_UP=1, RANGE=0, TREND_DN=-1 for scoring."""
    m = {"TREND_UP": 1.0, "RANGE": 0.0, "TREND_DN": -1.0}

    def cell(v: Any) -> float:
        s = str(v)
        return float(m.get(s, 0.0))

    return matrix.map(cell)


def confluence_score(
    matrix: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Weighted average directional score in [-1, 1] across timeframe columns."""
    if matrix.empty or matrix.shape[1] == 0:
        return pd.Series(0.0, index=matrix.index)
    num = regime_matrix_to_numeric(matrix)
    w = weights or {c: 1.0 for c in num.columns}
    ws = sum(w.get(c, 1.0) for c in num.columns)
    if ws <= 0:
        return pd.Series(0.0, index=matrix.index)
    acc = 0.0
    for c in num.columns:
        acc = acc + num[c] * float(w.get(c, 1.0))
    return acc / ws

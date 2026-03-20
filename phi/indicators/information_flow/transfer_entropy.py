"""Rolling transfer entropy indicators for multi-asset information flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def _extract_symbol_series(prices: pd.DataFrame, symbol: str) -> pd.Series:
    """Extract a symbol price series from flat or MultiIndex price tables."""
    if symbol in prices.columns:
        return prices[symbol].astype(float)

    if isinstance(prices.columns, pd.MultiIndex):
        if symbol in prices.columns.get_level_values(0):
            symbol_slice = prices[symbol]
            if isinstance(symbol_slice, pd.DataFrame) and "close" in symbol_slice.columns:
                return symbol_slice["close"].astype(float)
        if symbol in prices.columns.get_level_values(-1):
            if "close" in prices.columns.get_level_values(0):
                return prices[("close", symbol)].astype(float)

    if "close" in prices.columns and isinstance(prices["close"], pd.DataFrame) and symbol in prices["close"].columns:
        return prices["close"][symbol].astype(float)

    raise KeyError(f"Could not find symbol '{symbol}' in prices DataFrame.")


def _discretize(values: pd.Series, bins: int) -> np.ndarray:
    """Discretize returns into equal-frequency bins."""
    values = values.dropna()
    if values.empty:
        return np.array([], dtype=int)

    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values.to_numpy(dtype=float), q)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(values.size, dtype=int)

    codes = np.digitize(values.to_numpy(dtype=float), edges[1:-1], right=True)
    return codes.astype(int)


def _window_transfer_entropy(source_returns: pd.Series, target_returns: pd.Series, bins: int) -> float:
    """Estimate transfer entropy TE(source->target) on one aligned window."""
    aligned = pd.concat([source_returns, target_returns], axis=1, join="inner").dropna()
    if len(aligned) < 4:
        return np.nan

    src_disc = _discretize(aligned.iloc[:, 0], bins=bins)
    tgt_disc = _discretize(aligned.iloc[:, 1], bins=bins)
    n = min(len(src_disc), len(tgt_disc))
    if n < 4:
        return np.nan

    x_t = src_disc[:-1]
    y_t = tgt_disc[:-1]
    y_t1 = tgt_disc[1:]

    if len(y_t1) == 0:
        return np.nan

    counts_y1yx: dict[tuple[int, int, int], int] = {}
    counts_yx: dict[tuple[int, int], int] = {}
    counts_y1y: dict[tuple[int, int], int] = {}
    counts_y: dict[int, int] = {}

    for y1, y, x in zip(y_t1, y_t, x_t, strict=False):
        counts_y1yx[(int(y1), int(y), int(x))] = counts_y1yx.get((int(y1), int(y), int(x)), 0) + 1
        counts_yx[(int(y), int(x))] = counts_yx.get((int(y), int(x)), 0) + 1
        counts_y1y[(int(y1), int(y))] = counts_y1y.get((int(y1), int(y)), 0) + 1
        counts_y[int(y)] = counts_y.get(int(y), 0) + 1

    total = float(len(y_t1))
    te = 0.0
    for (y1, y, x), c_y1yx in counts_y1yx.items():
        p_y1yx = c_y1yx / total
        p_y1_given_yx = c_y1yx / (counts_yx[(y, x)] + _EPS)
        p_y1_given_y = counts_y1y[(y1, y)] / (counts_y[y] + _EPS)
        te += p_y1yx * np.log((p_y1_given_yx + _EPS) / (p_y1_given_y + _EPS))

    return float(max(te, 0.0))


def rolling_transfer_entropy(
    prices: pd.DataFrame,
    from_symbol: str,
    to_symbol: str,
    window: int = 50,
    bins: int = 3,
    normalize: bool = True,
) -> pd.Series:
    """Rolling transfer entropy from ``from_symbol`` to ``to_symbol``."""
    if window < 4:
        raise ValueError("window must be >= 4")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    source = _extract_symbol_series(prices, from_symbol)
    target = _extract_symbol_series(prices, to_symbol)
    source_returns = source.pct_change()
    target_returns = target.pct_change()

    out = pd.Series(np.nan, index=prices.index, dtype=float)
    for i in range(window - 1, len(prices)):
        start = i - window + 1
        src_w = source_returns.iloc[start : i + 1]
        tgt_w = target_returns.iloc[start : i + 1]
        te_val = _window_transfer_entropy(src_w, tgt_w, bins=bins)

        if normalize and np.isfinite(te_val):
            disc = _discretize(tgt_w.dropna(), bins=bins)
            if disc.size:
                _, counts = np.unique(disc, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + _EPS))
                if entropy > 0:
                    te_val = float(te_val / entropy)
        out.iloc[i] = te_val

    return out

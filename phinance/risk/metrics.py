"""Core portfolio risk metrics used by RL environments and runtime agents."""

from __future__ import annotations

import numpy as np


def compute_var_95(returns: np.ndarray) -> float:
    """Compute 1-day historical 95% VaR as a positive capital fraction."""
    if returns.size == 0:
        return 0.0
    q = float(np.quantile(returns, 0.05))
    return float(max(0.0, -q))


def compute_beta(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Compute beta relative to benchmark returns."""
    if returns.size == 0 or benchmark_returns.size == 0:
        return 0.0
    n = min(returns.size, benchmark_returns.size)
    asset = returns[-n:]
    bench = benchmark_returns[-n:]
    var_bench = float(np.var(bench))
    if var_bench <= 1e-12:
        return 0.0
    cov = float(np.cov(asset, bench)[0, 1])
    return cov / var_bench


def compute_correlation(matrix: np.ndarray) -> float:
    """Compute average pairwise correlation from an (n_assets, n_points) return matrix."""
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    corr = np.corrcoef(matrix)
    upper = corr[np.triu_indices_from(corr, k=1)]
    finite = upper[np.isfinite(upper)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))

"""Rolling KL-divergence based regime-shift indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-12


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(P || Q) for probability vectors."""
    return float(np.sum(p * np.log((p + _EPS) / (q + _EPS))))


def compute_kld_signal(
    ohlcv: pd.DataFrame,
    recent_window: int = 20,
    reference_window: int = 60,
    bins: int = 20,
    sigmoid_scale: float = 3.0,
) -> pd.Series:
    """Compare adjacent return distributions with symmetric KL divergence."""
    close = ohlcv.get("close")
    if close is None:
        logger.warning("KLD indicator requires 'close' column; returning zeros.")
        return pd.Series(0.0, index=ohlcv.index)

    returns = close.astype(float).pct_change()
    out = pd.Series(np.nan, index=ohlcv.index, dtype=float)

    min_needed = recent_window + reference_window
    if returns.notna().sum() < min_needed:
        logger.warning("KLD needs at least %s return points.", min_needed)

    for i in range(min_needed, len(returns) + 1):
        recent = returns.iloc[i - recent_window : i].dropna().values
        reference = returns.iloc[i - recent_window - reference_window : i - recent_window].dropna().values
        if recent.size < max(3, recent_window // 2) or reference.size < max(3, reference_window // 2):
            continue

        combined = np.concatenate([reference, recent])
        edges = np.histogram_bin_edges(combined, bins=bins)
        p_hist, _ = np.histogram(reference, bins=edges)
        q_hist, _ = np.histogram(recent, bins=edges)

        p = p_hist.astype(float) + _EPS
        q = q_hist.astype(float) + _EPS
        p /= p.sum()
        q /= q.sum()

        sym_kl = 0.5 * (_kl_divergence(p, q) + _kl_divergence(q, p))
        out.iloc[i - 1] = sym_kl

    score = np.tanh(out.fillna(0.0) / max(sigmoid_scale, 1e-6))
    return score.clip(-1.0, 1.0)

"""Rolling mutual information indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-12


def _mutual_information_from_hist(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Estimate normalized mutual information and map to [-1, 1]."""
    if x.size == 0 or y.size == 0:
        return 0.0
    joint, _, _ = np.histogram2d(x, y, bins=bins)
    total = joint.sum()
    if total <= 0:
        return 0.0

    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    valid = pxy > 0
    mi = np.sum(pxy[valid] * np.log((pxy[valid] + _EPS) / ((px @ py)[valid] + _EPS)))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0] + _EPS))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0] + _EPS))

    denom = max(min(hx, hy), _EPS)
    normalized = np.clip(mi / denom, 0.0, 1.0)
    return float(2.0 * (normalized - 0.5))


def compute_mutual_info_signal(
    ohlcv: pd.DataFrame,
    window: int = 20,
    bins: int = 20,
    mode: str = "price_volume",
) -> pd.Series:
    """Compute rolling mutual information signal in [-1, 1]."""
    if window < 3:
        logger.warning("Mutual information window too short (%s); using 3.", window)
        window = 3

    close = ohlcv.get("close")
    if close is None:
        logger.warning("Mutual information requires 'close' column; returning zeros.")
        return pd.Series(0.0, index=ohlcv.index)

    returns = close.astype(float).pct_change()
    if mode == "returns":
        x = returns
        y = returns.shift(1)
    else:
        volume = ohlcv.get("volume")
        if volume is None:
            logger.warning("Mutual information mode price_volume requires 'volume'; falling back to returns mode.")
            x = returns
            y = returns.shift(1)
        else:
            x = returns
            y = volume.astype(float).pct_change()

    out = pd.Series(np.nan, index=ohlcv.index, dtype=float)
    for i in range(window, len(ohlcv) + 1):
        xw = x.iloc[i - window : i]
        yw = y.iloc[i - window : i]
        aligned = pd.concat([xw, yw], axis=1).dropna()
        if len(aligned) < max(3, window // 2):
            continue
        out.iloc[i - 1] = _mutual_information_from_hist(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values, bins=bins)

    return out.fillna(0.0).clip(-1.0, 1.0)

"""Signal generation utilities derived from simplified MFT field properties."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.logging import get_logger
from phi.mft.field import compute_field_dynamics

logger = get_logger(__name__)


def mft_signal(
    close: pd.Series,
    kernel: str = "gaussian",
    sigma: float = 10.0,
    threshold: float = 0.0,
    smooth_window: int = 1,
) -> pd.Series:
    """Generate an MFT directional signal from the potential gradient.

    Signal logic:
      1) Compute potential and gradient.
      2) Signal is sign(gradient) when |gradient| > threshold else 0.
      3) Optional moving-average smoothing over the final signal.
    """
    if threshold < 0:
        logger.warning("Negative threshold=%s provided; using absolute value", threshold)
        threshold = abs(threshold)

    dyn = compute_field_dynamics(series=close, kernel=kernel, sigma=sigma)
    grad = dyn["gradient"]
    raw = pd.Series(np.where(grad.abs() > threshold, np.sign(grad), 0.0), index=close.index)

    if smooth_window > 1:
        raw = raw.rolling(int(smooth_window), min_periods=1).mean()
    return raw.clip(-1.0, 1.0).rename("mft_signal")


def mft_energy_signal(
    close: pd.Series,
    kernel: str = "gaussian",
    sigma: float = 10.0,
    energy_window: int = 20,
) -> pd.Series:
    """Generate a mean-reversion style signal from relative field energy.

    Higher-than-normal energy maps to more negative values, while low energy
    maps to more positive values.
    """
    dyn = compute_field_dynamics(series=close, kernel=kernel, sigma=sigma)
    energy = dyn["energy"]
    baseline = energy.rolling(int(energy_window), min_periods=max(1, int(energy_window) // 3)).mean()
    ratio = (energy / baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    signal = np.tanh(-(ratio - 1.0))
    return pd.Series(signal, index=close.index, name="mft_energy_signal").clip(-1.0, 1.0)

"""Helper utilities for simplified Market Field Theory computations."""

from __future__ import annotations

import numpy as np

from phi.logging import get_logger

logger = get_logger(__name__)


def build_kernel(kernel: str = "gaussian", sigma: float = 10.0) -> np.ndarray:
    """Build a normalized 1D kernel for temporal field convolution.

    Args:
        kernel: Kernel family ("gaussian", "exp", "linear").
        sigma: Width / decay control. Must be positive.

    Returns:
        Normalized kernel that sums to 1.
    """
    if sigma <= 0:
        logger.warning("Invalid sigma=%s; using fallback sigma=1.0", sigma)
        sigma = 1.0

    kernel_name = str(kernel).lower().strip()
    if kernel_name == "gaussian":
        radius = max(1, int(np.ceil(3 * sigma)))
        t = np.arange(-radius, radius + 1, dtype=float)
        w = np.exp(-(t**2) / (2.0 * sigma**2))
    elif kernel_name in {"exp", "exponential"}:
        radius = max(1, int(np.ceil(5 * sigma)))
        t = np.arange(-radius, radius + 1, dtype=float)
        w = np.exp(-np.abs(t) / sigma)
    elif kernel_name in {"linear", "triangular"}:
        radius = max(1, int(np.ceil(sigma)))
        t = np.arange(-radius, radius + 1, dtype=float)
        w = np.clip(1.0 - np.abs(t) / sigma, a_min=0.0, a_max=None)
    else:
        logger.warning("Unknown MFT kernel '%s'; falling back to gaussian", kernel)
        return build_kernel(kernel="gaussian", sigma=sigma)

    total = float(w.sum())
    if total <= 0:
        logger.warning("Degenerate kernel for kernel=%s sigma=%s; using identity kernel", kernel_name, sigma)
        return np.array([1.0], dtype=float)
    return (w / total).astype(float)

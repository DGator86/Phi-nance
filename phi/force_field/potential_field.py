"""1-D Gaussian liquidity/gamma wells and numerical negative gradient (force proxy)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GaussianWell:
    center: float
    depth: float
    sigma: float

    def potential(self, x: float) -> float:
        if self.sigma <= 0.0:
            return 0.0
        z = (x - self.center) / self.sigma
        return -self.depth * math.exp(-0.5 * z * z)


def total_potential(x: float, wells: list[GaussianWell]) -> float:
    return sum(w.potential(x) for w in wells)


def neg_gradient_1d(
    x: float,
    wells: list[GaussianWell],
    h: float | None = None,
) -> float:
    """F(x) ≈ -dU/dx via central difference."""
    if not wells:
        return 0.0
    base_h = 1e-4 * max(1.0, abs(x))
    step = h if h is not None else base_h
    up = total_potential(x + step, wells)
    dn = total_potential(x - step, wells)
    return -(up - dn) / (2.0 * step)

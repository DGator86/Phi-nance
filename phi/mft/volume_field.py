"""Volume-as-field utilities for advanced Market Field Theory (MFT)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.mft.field import field_gradient, field_potential


def volume_price_interaction(
    price_series: pd.Series,
    volume_series: pd.Series,
    kernel: str = "gaussian",
    sigma: float = 10.0,
    corr_window: int = 20,
) -> pd.Series:
    """Compute price/volume field interaction strength.

    The interaction combines (a) potential-product alignment, (b) gradient coupling,
    and (c) rolling correlation between field potentials.
    """
    price_potential = field_potential(price_series, kernel=kernel, sigma=sigma)
    volume_potential = field_potential(volume_series, kernel=kernel, sigma=sigma)

    price_gradient = field_gradient(price_potential)
    volume_gradient = field_gradient(volume_potential)

    product_term = price_potential * volume_potential
    gradient_term = price_gradient * volume_gradient
    correlation_term = price_potential.rolling(int(corr_window), min_periods=max(5, int(corr_window) // 2)).corr(volume_potential)

    interaction = (
        product_term / (product_term.abs().rolling(int(corr_window), min_periods=1).mean().replace(0.0, np.nan))
        + gradient_term
        + correlation_term.fillna(0.0)
    ) / 3.0

    return interaction.replace([np.inf, -np.inf], np.nan).fillna(0.0).rename("mft_price_volume_interaction")

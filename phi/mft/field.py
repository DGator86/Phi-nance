"""Core field computations for a simplified Market Field Theory (MFT) model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phi.mft.utils import build_kernel


def field_potential(series: pd.Series, kernel: str = "gaussian", sigma: float = 10.0) -> pd.Series:
    """Compute field potential by convolving the input series with a kernel."""
    values = series.astype(float).to_numpy()
    valid = pd.Series(values, index=series.index).ffill().bfill().fillna(0.0).to_numpy()
    weights = build_kernel(kernel=kernel, sigma=sigma)

    pad = len(weights) // 2
    padded = np.pad(valid, (pad, pad), mode="edge") if pad > 0 else valid
    potential = np.convolve(padded, weights, mode="valid")
    return pd.Series(potential, index=series.index, name="potential")


def field_gradient(potential: pd.Series) -> pd.Series:
    """First-order difference of field potential (force proxy)."""
    gradient = potential.astype(float).diff().fillna(0.0)
    gradient.name = "gradient"
    return gradient


def field_laplacian(potential: pd.Series) -> pd.Series:
    """Second-order difference of field potential (curvature proxy)."""
    laplacian = potential.astype(float).diff().diff().fillna(0.0)
    laplacian.name = "laplacian"
    return laplacian


def field_energy(gradient: pd.Series) -> pd.Series:
    """Squared gradient as a simple field-energy / activity proxy."""
    energy = gradient.astype(float).pow(2)
    energy.name = "energy"
    return energy


def compute_field_dynamics(series: pd.Series, kernel: str = "gaussian", sigma: float = 10.0) -> pd.DataFrame:
    """Compute potential and first/second-order derived MFT quantities."""
    potential = field_potential(series=series, kernel=kernel, sigma=sigma)
    gradient = field_gradient(potential)
    laplacian = field_laplacian(potential)
    energy = field_energy(gradient)
    return pd.DataFrame(
        {
            "potential": potential,
            "gradient": gradient,
            "laplacian": laplacian,
            "energy": energy,
        },
        index=series.index,
    )

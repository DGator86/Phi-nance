from __future__ import annotations

import numpy as np
import pandas as pd

from phi.mft.field import compute_field_dynamics, field_potential
from phi.mft.signals import mft_energy_signal, mft_signal


def test_field_potential_preserves_shape_and_index() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)

    potential = field_potential(close, kernel="gaussian", sigma=8.0)

    assert isinstance(potential, pd.Series)
    assert len(potential) == len(close)
    assert potential.index.equals(close.index)
    assert potential.notna().all()


def test_compute_field_dynamics_columns_and_lengths() -> None:
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    close = pd.Series(100.0 + np.sin(np.linspace(0, 6, len(idx))), index=idx)

    dynamics = compute_field_dynamics(close, kernel="exp", sigma=6.0)

    assert set(["potential", "gradient", "laplacian", "energy"]).issubset(dynamics.columns)
    assert len(dynamics) == len(close)
    assert dynamics.index.equals(close.index)


def test_mft_signal_detects_direction_on_monotonic_series() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    up_close = pd.Series(np.linspace(10.0, 30.0, len(idx)), index=idx)
    down_close = pd.Series(np.linspace(30.0, 10.0, len(idx)), index=idx)

    up_signal = mft_signal(up_close, kernel="gaussian", sigma=4.0, threshold=0.0, smooth_window=1)
    down_signal = mft_signal(down_close, kernel="gaussian", sigma=4.0, threshold=0.0, smooth_window=1)

    assert up_signal.iloc[-10:].mean() > 0
    assert down_signal.iloc[-10:].mean() < 0


def test_mft_energy_signal_bounded_range() -> None:
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    close = pd.Series(100 + np.sin(np.linspace(0, 4 * np.pi, len(idx))), index=idx)

    signal = mft_energy_signal(close, sigma=5.0, energy_window=15)

    assert signal.between(-1.0, 1.0).all()

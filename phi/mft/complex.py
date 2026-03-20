"""Complex-valued Market Field Theory (MFT) features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import hilbert


def hilbert_transform(series: pd.Series) -> pd.Series:
    """Return non-causal Hilbert transform (imaginary part) of a real-valued series."""
    real = series.astype(float)
    filled = real.ffill().bfill().fillna(0.0)
    analytic = hilbert(filled.to_numpy())
    imag = pd.Series(np.imag(analytic), index=series.index, name="imag")
    return imag.where(real.notna())


def hilbert_transform_rolling(series: pd.Series, window: int = 50) -> pd.Series:
    """Return causal Hilbert transform (imaginary part) via trailing windows."""
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")

    real = series.astype(float)
    filled = real.ffill().bfill().fillna(0.0)
    out = pd.Series(np.nan, index=series.index, name="imag", dtype=float)

    for i in range(w - 1, len(filled)):
        window_data = filled.iloc[i - w + 1 : i + 1].to_numpy(dtype=float)
        analytic = hilbert(window_data)
        out.iloc[i] = float(np.imag(analytic[-1]))

    return out.where(real.notna())


def complex_potential(series: pd.Series, hilbert_window: int = 50) -> pd.DataFrame:
    """Return analytic-signal components and derived complex potential features.

    Columns:
      - ``real``: input real-valued series.
      - ``imag``: Hilbert-transform component (quadrature pair).
      - ``amplitude``: instantaneous magnitude ``sqrt(real^2 + imag^2)``.
      - ``phase``: instantaneous phase angle in radians.
      - ``phase_change``: unwrapped phase first difference (frequency proxy).
    """
    real = series.astype(float)
    imag = hilbert_transform_rolling(real, window=hilbert_window)

    analytic = real.to_numpy(dtype=float) + 1j * imag.fillna(0.0).to_numpy(dtype=float)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    phase_series = pd.Series(phase, index=series.index)
    valid = real.notna() & imag.notna()
    phase_unwrapped = phase_series.copy()
    segment_ids = valid.ne(valid.shift(fill_value=False)).cumsum()
    for segment_id in segment_ids[valid].unique():
        mask = (segment_ids == segment_id) & valid
        if int(mask.sum()) > 1:
            phase_unwrapped.loc[mask] = np.unwrap(phase_series.loc[mask].to_numpy(dtype=float))

    phase_change = phase_unwrapped.diff().fillna(0.0).where(valid)

    frame = pd.DataFrame(
        {
            "real": real,
            "imag": imag,
            "amplitude": pd.Series(amplitude, index=series.index),
            "phase": phase_series,
            "phase_change": phase_change,
        },
        index=series.index,
    )
    return frame.where(real.notna())

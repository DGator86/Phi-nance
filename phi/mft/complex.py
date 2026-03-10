"""Complex-valued Market Field Theory (MFT) features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import hilbert


def hilbert_transform(series: pd.Series) -> pd.Series:
    """Return the Hilbert transform (imaginary part) of a real-valued series."""
    real = series.astype(float)
    filled = real.ffill().bfill().fillna(0.0)
    analytic = hilbert(filled.to_numpy())
    imag = pd.Series(np.imag(analytic), index=series.index, name="imag")
    return imag.where(real.notna())


def complex_potential(series: pd.Series) -> pd.DataFrame:
    """Return analytic-signal components and derived complex potential features.

    Columns:
      - ``real``: input real-valued series.
      - ``imag``: Hilbert-transform component (quadrature pair).
      - ``amplitude``: instantaneous magnitude ``sqrt(real^2 + imag^2)``.
      - ``phase``: instantaneous phase angle in radians.
      - ``phase_change``: unwrapped phase first difference (frequency proxy).
    """
    real = series.astype(float)
    imag = hilbert_transform(real)

    analytic = real.to_numpy(dtype=float) + 1j * imag.fillna(0.0).to_numpy(dtype=float)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    phase_unwrapped = np.unwrap(phase)
    phase_change = np.diff(phase_unwrapped, prepend=phase_unwrapped[0])

    frame = pd.DataFrame(
        {
            "real": real,
            "imag": imag,
            "amplitude": pd.Series(amplitude, index=series.index),
            "phase": pd.Series(phase, index=series.index),
            "phase_change": pd.Series(phase_change, index=series.index),
        },
        index=series.index,
    )
    return frame.where(real.notna())

# Advanced Market Field Theory (MFT)

This document describes the advanced MFT indicators added on top of the baseline potential/gradient implementation.

## 1) Complex potentials via Hilbert transform

Price can be represented as an analytic signal:

- Real part: original series.
- Imaginary part: Hilbert transform (quadrature component).

From this representation we derive:

- **Amplitude**: local oscillation intensity.
- **Phase**: cycle position in radians.
- **Phase change**: first difference of unwrapped phase, a proxy for instantaneous frequency changes.

Use when you want to detect cyclic structure shifts that are not obvious in raw price returns.

## 2) Volume as a field

Volume is treated as an independent scalar field and smoothed into a potential the same way as price.

Interaction terms combine:

- price potential × volume potential,
- gradient coupling,
- rolling correlation between price and volume potentials.

A positive interaction generally implies aligned price/volume dynamics; weaker or negative interaction may imply divergence.

## 3) Fourier-domain indicators

Rolling FFT over a fixed window gives local spectrum estimates.

Current advanced indicator exposes **relative spectral band power** using band limits as fractions of Nyquist:

- low: `[0.0, 0.2)`
- mid: `[0.2, 0.5)`
- high: `[0.5, 1.0)`

This helps differentiate trend-like low-frequency regimes from noisy high-frequency regimes.

## 4) Usage examples

```python
from phi.mft.complex import complex_potential
from phi.mft.volume_field import volume_price_interaction
from phi.mft.fourier import rolling_spectral_power

cp = complex_potential(close)
amp = cp["amplitude"]
phase = cp["phase"]

pvi = volume_price_interaction(close, volume, kernel="gaussian", sigma=10.0, corr_window=20)

bands = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
power = rolling_spectral_power(close, window=64, bands=bands)
```

## 5) Parameter guidance

- **sigma** (field smoothing):
  - lower values react faster but are noisier,
  - higher values are more stable but lag more.
- **corr_window** (price-volume interaction):
  - 10-30 for swing-scale,
  - 30+ for slower regime diagnostics.
- **FFT window**:
  - 32-64 for faster adaptation,
  - 64-128 for more stable spectral estimates.

## 6) Streamlit indicators

New indicators available under **Market Field Theory**:

- MFT Complex Amplitude
- MFT Complex Phase
- MFT Phase Change
- MFT Price-Volume Interaction
- MFT Spectral Power

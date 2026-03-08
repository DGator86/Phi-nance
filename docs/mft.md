# Market Field Theory (MFT) — Simplified Implementation

## Overview

This project includes an **experimental** Market Field Theory (MFT) implementation in `phi/mft/`.

The core idea is to treat price as a 1D temporal field:

1. Build a smoothed **field potential** from close prices via kernel convolution.
2. Derive local dynamics from the potential:
   - **Gradient** (first difference): directional force proxy.
   - **Laplacian** (second difference): curvature / turning-pressure proxy.
   - **Energy** (`gradient^2`): activity or volatility proxy.
3. Convert these into bounded trading signals in `[-1, 1]`.

> This is intentionally interpretable and lightweight, not a strict physics derivation.

## Components

- `phi/mft/utils.py`
  - `build_kernel(kernel, sigma)`
  - Supported kernels:
    - `gaussian`: `exp(-t^2 / (2*sigma^2))`
    - `exp`: `exp(-|t| / sigma)`
    - `linear`: triangular kernel

- `phi/mft/field.py`
  - `field_potential(series, kernel, sigma)`
  - `field_gradient(potential)`
  - `field_laplacian(potential)`
  - `field_energy(gradient)`
  - `compute_field_dynamics(series, kernel, sigma)`

- `phi/mft/signals.py`
  - `mft_signal(close, kernel, sigma, threshold, smooth_window)`
  - `mft_energy_signal(close, kernel, sigma, energy_window)`

## Indicator Integration

Two MFT indicators are available in the Streamlit workbench under **Market Field Theory**:

- **MFT Signal**
  - Signal from gradient direction with threshold filter.
  - Parameters:
    - `kernel`: `gaussian | exp | linear`
    - `sigma`: kernel width/decay
    - `threshold`: minimum absolute gradient to emit non-zero signal
    - `smooth_window`: optional smoothing on final signal

- **MFT Energy**
  - Signal from relative energy regime (energy vs rolling baseline).
  - Parameters:
    - `kernel`
    - `sigma`
    - `energy_window`

## Interpretation Notes

- Positive **MFT Signal** implies upward local field force.
- Negative **MFT Signal** implies downward local field force.
- **MFT Energy** can help filter regimes:
  - Very high energy: unstable / reactive environments.
  - Lower energy: calmer environments.

## Practical Usage

Start simple:

- `kernel = gaussian`
- `sigma = 10`
- `threshold = 0.0`
- `smooth_window = 3`

Then tune with backtests by market and timeframe.

## Caveats

- This feature is **experimental** and should be validated empirically.
- Results are sensitive to kernel shape and `sigma`.
- Signals are heuristic and may evolve as MFT research in the project matures.


## Backward compatibility

- Registry key `phi_mft` is retained as an alias to the simplified `mft_signal` computation.
- Simple-indicator name `Phi-Bot (MFT)` is retained as an alias to `MFT Signal`.

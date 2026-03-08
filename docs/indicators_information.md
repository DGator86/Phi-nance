# Information Theory Indicators

This document describes information-theoretic indicators available in `phi.indicators`.

## Why information theory?

Information theory helps quantify:

- **Uncertainty / complexity** in price movement.
- **Dependency / information transfer** between variables (for example, returns and volume).
- **Distribution shifts** that may indicate a market regime change.

All indicators in this module are rolling and return normalized signals in `[-1, 1]` for blend compatibility.

## Indicators

## 1) Return Entropy (`return_entropy`)

- Implementation: `phi/indicators/information/entropy.py`
- Core idea: compute Shannon entropy of returns histogram in a rolling window.
- Parameters:
  - `window` (default `20`)
  - `bins` (default `20`)
  - `base` (default `2`)

### Normalization

1. Entropy is computed as `H = -Σ p log_base(p)`.
2. It is divided by `log_base(n_bins)` to map to `[0, 1]`.
3. Final signal maps to `[-1, 1]` via `2 * (norm - 0.5)`.

Interpretation:

- Higher values => more disorder/choppiness.
- Lower values => more predictable/trending behavior.

## 2) Mutual Information (`mutual_information`)

- Implementation: `phi/indicators/information/mutual_info.py`
- Core idea: estimate MI between returns and volume changes (or between successive returns).
- Parameters:
  - `window` (default `20`)
  - `bins` (default `20`)
  - `mode` (`price_volume` or `returns`, default `price_volume`)

### Normalization

1. Estimate MI from joint + marginal histograms.
2. Normalize by `min(Hx, Hy)` to approximately scale to `[0, 1]`.
3. Convert to `[-1, 1]` with `2 * (norm - 0.5)`.

Interpretation:

- Higher values => stronger dependency/informed-flow behavior.
- Lower values => decoupling/noisy behavior.

## 3) Fisher Information Proxy (`fisher_information`)

- Implementation: `phi/indicators/information/fisher.py`
- Practical proxy: inverse rolling return variance (`1 / var`).
- Parameters:
  - `window` (default `20`)
  - `clip_percentile` (default `95`)

### Normalization

- Proxy values are clipped at chosen percentile to avoid outliers.
- Clipped values are scaled by percentile cap into `[0, 1]`.
- Converted to `[-1, 1]` with `2 * (norm - 0.5)`.

Interpretation:

- Higher => lower variance / more stable regime.
- Lower => higher variance / unstable regime.

## 4) KL Divergence (`kld_regime_shift`)

- Implementation: `phi/indicators/information/kld.py`
- Compares two adjacent windows of returns:
  - reference (older)
  - recent (newer)
- Uses **symmetric KL**: `0.5 * (KL(P||Q) + KL(Q||P))`.
- Parameters:
  - `recent_window` (default `20`)
  - `reference_window` (default `60`)
  - `bins` (default `20`)
  - `sigmoid_scale` (default `3.0`)

### Normalization

- Symmetric KL is non-negative and unbounded.
- Signal uses `tanh(kld / sigmoid_scale)` to map into `[-1, 1]`.

Interpretation:

- Higher => stronger distribution shift / potential regime transition.
- Lower => distribution relatively stable.

## Streamlit integration

Indicators are exposed in the Streamlit workbench under **Information Theory**:

- Return Entropy
- Mutual Information
- Fisher Information
- KL Divergence

Each indicator includes parameter controls that map directly to compute arguments.

## Strategy usage examples

- Use `kld_regime_shift` as a **regime-change gate** for trend systems.
- Use high `return_entropy` periods to **reduce position size**.
- Blend `mutual_information` with order-flow signals to detect **informed participation**.
- Prefer trend-following signals when `fisher_information` is high (stable regimes).

## Performance and caveats

- Histogram binning can materially change signal behavior.
- Rolling custom computations are O(n * window); larger windows increase runtime.
- Missing/invalid `volume` for MI (`price_volume`) falls back to returns mode with a warning.
- For very short history windows, indicators return zeros for unavailable warmup regions.

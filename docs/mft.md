# Market Field Theory (MFT) Notes

This document summarizes the MFT framing used by Phi-nance regime detection.

## Core concepts

- **Field strength**: directional pressure inferred from price trend, momentum, and breadth-like proxies.
- **Entropy**: uncertainty/disorder of recent returns, approximated with rolling distribution metrics.
- **Fractal dimension**: roughness of the price path, used to discriminate trend persistence from noisy mean reversion.

## Feature computation

Typical MFT-inspired features are computed on rolling windows over OHLCV:

1. Return, volatility, and drawdown transforms.
2. Entropy-like statistics from return bins or normalized volatility changes.
3. Regime persistence and transition scores.
4. Optional volume/participation features.

## Regime classifier usage

The classifier combines these features into labels such as:

- `TREND_UP` / `TREND_DN`
- `RANGE`
- `BREAKOUT_UP` / `BREAKOUT_DN`
- `HIGHVOL` / `LOWVOL`

Those labels can be converted to probabilities and consumed by blending (`regime_weighted`) to adapt signal weights.

## References

- Peters, E. E. *Fractal Market Analysis*.
- Mandelbrot, B. *The (Mis)Behavior of Markets*.
- Related literature on entropy in financial time series and regime-switching models.

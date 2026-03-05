# Options Trading Instructions

- Use deterministic pricing inputs in tests (`S, K, T, r, sigma`) and include reference values.
- For new pricing models, include:
  - one benchmark test,
  - one edge-case test,
  - one model-consistency test.
- Greeks must always expose `delta`, `gamma`, `theta`, `vega`, and `rho`.
- Keep model functions pure (no I/O inside pricing functions).
- Data-fetching code must cache vendor responses and provide a cache-read helper.

- IV surfaces should clamp extrapolated queries to the nearest valid strike/expiry bounds.
- Backtests should return portfolio value series, aggregate Greeks, and a trade log.

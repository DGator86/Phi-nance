# PhiAI

PhiAI orchestrates parameter search and optimization workflows for strategy tuning.

## Capabilities

- Indicator parameter tuning
- Walk-forward style evaluation support
- Parallel trial execution controls via env vars
- Explanation output for selected configurations

## Core configuration

- `PHIAI_DEFAULT_N_TRIALS`
- `PHIAI_PARALLEL_JOBS`
- `PHIAI_WALK_FORWARD_WINDOWS`

## Typical flow

1. Build dataset and indicator config.
2. Run PhiAI optimization.
3. Review best parameters and explanation.
4. Re-run backtest with optimized configuration.

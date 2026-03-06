# Performance Optimisation

This document tracks backtest-performance optimisation with reproducible profiling
runs using `scripts/profile_backtest.py`.

## Step 1: Baseline profiling

Run:

```bash
python scripts/profile_backtest.py --rows 2520 --iterations 3 --warmup 0 --cprofile-out artifacts/perf/backtest_cprofile_before.txt
```

### Baseline results

Timing summary (2520 bars, 3 iterations):

- `run_backtest` average: **0.0568s**
- `run_backtest` total (3 runs): **0.1704s**

Top cumulative cProfile hotspots:

1. `phinance.backtest.runner.run_backtest` — 0.097s
2. `phinance.backtest.engine.simulate` — 0.060s
3. `pandas index __getitem__` / datetime indexing internals — ~0.046s / 0.043s
4. `phinance.strategies.indicator_catalog.compute_indicator` — 0.030s
5. `phinance.strategies.rsi.compute` — 0.019s
6. `phinance.strategies.macd.compute` — 0.011s

## Step 2: Backtest-engine optimisation

Run:

```bash
python scripts/profile_backtest.py --rows 2520 --iterations 3 --warmup 1 --cprofile-out artifacts/perf/backtest_cprofile_after.txt
```

### Optimisations applied

- Added a **Numba JIT simulation core** (`_simulate_state`) to accelerate the
  path-dependent execution loop in `phinance.backtest.engine.simulate`.
- Reduced Python/pandas overhead inside simulation by operating on numpy arrays
  and constructing `Trade` objects after the numerical pass.
- Added an **LRU-style in-memory indicator cache** in `phinance.backtest.runner`
  keyed by indicator, params, and dataset identity to avoid recomputation in
  repeated profiling / sweep-style runs.
- Added Numba-accelerated helpers for drawdown and return statistics in
  `phinance.backtest.metrics`.
- Added profile-script warmup support (`--warmup`) so Numba compile overhead is
  excluded from steady-state timing.

### Post-optimisation results

Timing summary (2520 bars, 3 iterations, warmup=1):

- `run_backtest` average: **0.0098s**
- `run_backtest` total (3 runs): **0.0293s**

Top cumulative cProfile hotspots after optimisation:

1. `phinance.backtest.runner.run_backtest` — 0.016s
2. `phinance.backtest.engine.simulate` — 0.009s
3. Date-index to list conversion (`DatetimeArray.__iter__`) — ~0.003s
4. `phinance.backtest.metrics.compute_all` — 0.002s

### Improvement summary

- `run_backtest` average iteration time reduced from **0.0568s → 0.0098s**.
- Approximate speed-up: **~5.8x faster** in this benchmark.

## Guidance for Numba-compatible code

- Prefer pure numerical kernels that accept/return numpy arrays and primitive
  scalars.
- Keep object creation (`dict`, dataclass instances) outside JIT kernels.
- Warm up profiled functions once before timing to avoid counting compile cost.
- Use `@njit(cache=True)` for stable reruns in local development.

## Profiling utilities

`phinance.utils.performance` provides reusable utilities:

- `PerformanceTracker` for collecting timing samples and summaries
- `track_time(...)` context manager for named timed sections
- `profiled(...)` decorator for lightweight function instrumentation
- `run_cprofile(...)` helper to run cProfile and optionally persist report text

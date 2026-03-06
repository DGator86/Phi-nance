# Performance Optimisation

This document starts Part A of the optimisation roadmap with a reproducible
baseline profile for the backtest pipeline.

## Step 1: Baseline profiling

Use the profiling script to generate timing summaries and a cProfile report for
`phinance.backtest.runner.run_backtest`.

```bash
source venv/bin/activate
python scripts/profile_backtest.py --rows 2520 --iterations 3
```

Optional config values live in `configs/optimisation_config.yaml`.

## What gets measured

The script records:

- Dataset preparation time (`prepare_dataset`)
- End-to-end backtest execution time (`run_backtest`) over repeated iterations
- Function-level hotspots via cProfile (`artifacts/perf/backtest_cprofile.txt`)

## Profiling utilities

`phinance.utils.performance` provides reusable utilities:

- `PerformanceTracker` for collecting timing samples and summaries
- `track_time(...)` context manager for named timed sections
- `profiled(...)` decorator for lightweight function instrumentation
- `run_cprofile(...)` helper to run cProfile and optionally persist report text

## Next steps

After generating baseline numbers:

1. Identify top cumulative-time functions from the cProfile report.
2. Prioritise vectorisation/JIT opportunities in signal generation and simulation.
3. Re-run this baseline script after each optimisation and track deltas over time.

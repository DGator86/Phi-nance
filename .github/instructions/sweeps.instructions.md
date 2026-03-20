# Sweeps implementation instructions

Use `phinance.experiment.sweep.SweepRunner` for all sweep orchestration.

## Requirements

- Sweep configs should be YAML and include:
  - base experiment fields (`name`, `target`, `tracking`, etc.)
  - `search_space` keyed by dotted parameter paths
  - `sweep.method` as `grid` or `random`
- Keep compatibility with single-run experiment configs.
- Link child runs to parent run with `mlflow.parentRunId` tag.
- Log sweep summary metrics and trial table artifacts.

## Recommended defaults

- `sweep.parallel: 1`
- `sweep.method: grid`
- `tracking.backend: none` for local smoke tests

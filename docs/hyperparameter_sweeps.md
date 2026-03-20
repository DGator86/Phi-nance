# Hyperparameter Sweeps

Phi-nance supports systematic sweep execution on top of the existing experiment runner.

## Sweep config format

Sweep configs are YAML files that include the normal experiment fields plus `search_space` and `sweep`.

```yaml
name: execution_agent_sweep
tracking:
  backend: mlflow
  uri: ./mlruns
target: train_execution_agent
params:
  learning_rate: 0.0003

search_space:
  params.learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-2
  params.batch_size:
    type: choice
    values: [32, 64, 128]

sweep:
  method: random  # grid | random
  n_trials: 20    # random only
  parallel: 4
  objective_metric: sharpe
  objective_mode: max
```

Supported parameter types:
- `choice`: picks from `values`
- `int`: integer range from `low` to `high` (inclusive)
- `uniform`: float in [`low`, `high`]
- `loguniform`: exponential-uniform between positive bounds

For `grid` sweeps, `choice` and `int` are enumerated automatically. Continuous types (`uniform`, `loguniform`) require explicit `values` when used with `grid`.

## Running a sweep

```bash
python scripts/run_sweep.py --config configs/sweeps/gp_discovery_sweep.yaml
```

The sweep runner:
1. Creates a parent tracking run.
2. Expands/samples trial configs.
3. Executes each trial via `run_experiment`.
4. Logs summary metrics and a CSV artifact of all trial results.

## Viewing results in MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Trials are tagged with `mlflow.parentRunId`, `sweep.name`, and `sweep.trial_index`.

## Backward compatibility

If a config does not define `search_space`, it is treated as a single-point sweep and runs exactly one trial.

## Extending search types

Add a new type in `phinance/experiment/search_space.py` by:
- validating required keys in `_validate_space`
- implementing sampling/enumeration in `_sample_for_spec` and `_grid_values_for_spec`

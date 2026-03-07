# Experimentation in Phi-nance

This project now includes an opt-in experiment framework under `phinance.experiment`.

## Install dependencies

```bash
pip install -r requirements.txt
# or
pip install .[experiment]
```

## Config format

Create YAML configs under `configs/experiments/`.

Required fields:
- `name`
- `target`

Optional sections:
- `description`
- `tracking` (`backend`: `mlflow` | `wandb` | `none`, `uri`)
- `params`
- `data`
- `tags`

## Run an experiment

```bash
python scripts/run_experiment.py --config configs/experiments/train_execution_agent.yaml
```

With overrides:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/train_execution_agent.yaml \
  --override params.fallback=true \
  --override tracking.backend=mlflow
```

## MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Compare results

```bash
python -m phinance.experiment.results --experiment-name execution_agent_test
python -m phinance.experiment.results --compare <run_id_1> <run_id_2>
```

## Reproducibility captured

Each run logs:
- parameters and data-section values
- git commit hash and dirty-working-tree flag (tags)
- `pip freeze` snapshot as an artifact

## Extending targets

Register a short alias in `TARGET_REGISTRY` or pass a fully-qualified target:

```yaml
target: "my_package.my_module:my_function"
```

Target callable contract:

```python
def my_function(..., tracker=None) -> dict[str, float]:
    ...
    return {"metric": 1.0}
```

Existing scripts remain backward compatible because `tracker` is optional.


## Hyperparameter sweeps

Use sweep configs under `configs/sweeps/` and run:

```bash
python scripts/run_sweep.py --config configs/sweeps/gp_discovery_sweep.yaml
```

See `docs/hyperparameter_sweeps.md` for config schema, search-space types, and MLflow sweep analysis workflow.

## Interactive Analysis with Jupyter Notebooks

Install notebook dependencies:

```bash
pip install -r requirements.txt
# or
pip install .[experiment,notebooks]
```

Launch Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```

Notebook templates live under `notebooks/`:

- `01_basic_analysis.ipynb`: inspect a single run, parameters, and learning curves.
- `02_sweep_analysis.ipynb`: compare child trials from a sweep and inspect parameter-performance patterns.
- `03_custom_analysis.ipynb`: scaffold for custom analysis workflows.

The notebooks use reusable helpers from `phinance.experiment.visualization` and run discovery from `phinance.experiment.results`. Set `MLFLOW_TRACKING_URI` (or configure it in code) before running cells.


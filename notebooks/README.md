# Experiment Analysis Notebooks

These notebooks provide a starting point for interactive MLflow analysis in Phi-nance.

## Setup

Install dependencies:

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

If needed, set your tracking URI before opening notebooks:

```bash
export MLFLOW_TRACKING_URI=./mlruns
```

## Notebook overview

- `01_basic_analysis.ipynb` – list runs, inspect params/metrics, plot a learning curve.
- `02_sweep_analysis.ipynb` – compare sweep trials, inspect parameter interactions, highlight top runs.
- `03_custom_analysis.ipynb` – template scaffold for custom ad-hoc research.

## Notes

- The notebooks rely on `phinance.experiment.results` and `phinance.experiment.visualization`.
- Ensure you have existing MLflow runs, otherwise notebook cells that fetch runs will be empty.

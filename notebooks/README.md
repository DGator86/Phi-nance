# Notebooks

## Phi-nance core (JupyterLab)

One-time setup from the **repository root** (venv activated):

```bash
pip install -r requirements-jupyter.txt
python -m ipykernel install --user --name phinance --display-name "Python (Phi-nance)"
```

In JupyterLab: **Kernel → Change Kernel → Python (Phi-nance)**.

Use **File → Open Folder** on the repo (or `jupyter lab /path/to/Phi-nance`). The bootstrap cell finds `notebook_setup.py` by walking up from the process working directory.

- **`00_getting_started.ipynb`** — path / `IS_BACKTESTING` / `.env`, `regime_engine`, `phinance`, `engine_health`.
- **`regime_engine/demo_notebook.ipynb`** — full MFT regime demo (matplotlib).

---

## Experiment / MLflow analysis

These notebooks support interactive MLflow analysis in Phi-nance.

### Setup

```bash
pip install -r requirements.txt
# or
pip install .[experiment,notebooks]
```

```bash
jupyter lab
# or
jupyter notebook
```

If needed:

```bash
export MLFLOW_TRACKING_URI=./mlruns
```

### Notebook overview

- `01_basic_analysis.ipynb` — list runs, inspect params/metrics, plot a learning curve.
- `02_sweep_analysis.ipynb` — compare sweep trials, parameter interactions, top runs.
- `03_custom_analysis.ipynb` — template for ad-hoc research.

Rely on `phinance.experiment.results` and `phinance.experiment.visualization`. You need existing MLflow runs; otherwise run-fetching cells may be empty.

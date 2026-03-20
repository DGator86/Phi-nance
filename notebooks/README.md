# Phi-nance notebooks

## One-time setup

From the **repository root** (with your venv activated):

```bash
pip install -r requirements-jupyter.txt
python -m ipykernel install --user --name phinance --display-name "Python (Phi-nance)"
```

In JupyterLab: **Kernel → Change Kernel → Python (Phi-nance)**.

## Opening the project

Use **File → Open Folder** (or `jupyter lab /path/to/Phi-nance`) so the repo is the workspace. The bootstrap cell finds `notebook_setup.py` by walking up from the current working directory.

## Starter notebook

Open `00_getting_started.ipynb` and run all cells. The first cell loads path, `IS_BACKTESTING`, and `.env`.

The regime engine walkthrough also lives at `regime_engine/demo_notebook.ipynb`.

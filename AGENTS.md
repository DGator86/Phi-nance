# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Phi-nance is a quantitative trading platform built around a Market Field Theory (MFT) regime detection engine with a Streamlit dashboard UI. Single Python application (not a monorepo) with two internal packages: `regime_engine/` and `strategies/`.

### Running the app

```bash
source venv/bin/activate
streamlit run dashboard.py --server.headless true
```

Dashboard serves on **port 8501**. The `.streamlit/config.toml` binds to `0.0.0.0`.

### JupyterLab

```bash
source venv/bin/activate
pip install -r requirements-jupyter.txt
python -m ipykernel install --user --name phinance --display-name "Python (Phi-nance)"
jupyter lab
```

Open the **Phi-nance** folder as the Jupyter workspace. Run `notebook_setup.py` via the first cell in `notebooks/00_getting_started.ipynb` (or the same bootstrap block in `regime_engine/demo_notebook.ipynb`). That adds the repo to `sys.path`, sets `IS_BACKTESTING=True`, and loads `.env`.

Alternatively: `pip install -e ".[jupyter]"` if using the editable `pyproject.toml` install.

### Validating the engine

```bash
source venv/bin/activate
python engine_health.py
```

Runs the full MFT pipeline on synthetic OHLCV data (no API key needed). Exit code 0 = all 6 components pass.

### Key caveats

- **`IS_BACKTESTING` env var**: The dashboard auto-sets `IS_BACKTESTING=True` before importing lumibot. Without this, lumibot's `credentials.py` crashes trying to instantiate live brokers. This is already handled in `dashboard.py` line 34.
- **Tests**: `pytest` is configured under `tests/`; `engine_health.py` validates the MFT pipeline on synthetic data.
- **`.env` file**: Copy `.env.example` to `.env`. The default `AV_API_KEY` in `.env.example` is a free-tier Alpha Vantage key (rate-limited to 5 req/min). Backtests and data fetching require this key.
- **Optional services**: Ollama (for Plutus Bot tab) and Polygon.io (for L2 feed) are optional and the app gracefully degrades without them.
- **`python3.12-venv` system package**: Required to create the venv; install with `sudo apt-get install -y python3.12-venv` if not already present.

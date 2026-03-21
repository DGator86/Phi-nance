# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Phi-nance is a quantitative trading platform built around a Market Field Theory (MFT) regime detection engine with a Streamlit dashboard UI. Single Python application (not a monorepo) with two internal packages: `regime_engine/` and `strategies/`.

**Where code lives:** `docs/architecture_layout.md` (phi vs phinance vs app_streamlit vs legacy). **Branching:** `docs/DEV_WORKFLOW.md`.

### Running the app

**Default (easy mode — overview, automatic backtest, ticker spotlight):**

```bash
source venv/bin/activate
streamlit run app_streamlit/main.py --server.headless true
```

Same UI via `app_streamlit/live_workbench.py` (thin wrapper around `main`).

**Full workbench** (trading desk, regime tuning, LOB, live trading): `streamlit run app_streamlit/expert_workbench.py`.

See `docs/easy_mode.md` for env vars (`PHINANCE_UNIVERSE`, optional nightly learning JSON).

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
python scripts/engine_health.py
```

Runs the full MFT pipeline on synthetic OHLCV data (no API key needed). Exit code 0 = all 6 components pass.

### Key caveats

- **`IS_BACKTESTING` env var**: Set before importing lumibot (e.g. Streamlit entrypoints and `notebook_setup.py`). Without it, lumibot's `credentials.py` can crash when instantiating live brokers.
- **Tests**: `pytest` is configured under `tests/`; `python scripts/engine_health.py` validates the MFT pipeline on synthetic data.
- **`.env` file**: Copy `.env.example` to `.env`. The default `AV_API_KEY` in `.env.example` is a free-tier Alpha Vantage key (rate-limited to 5 req/min). Backtests and data fetching require this key.
- **Ecosystem OHLCV**: For Lumibot / TensorTrade / agent-cli glue, use `phi.data.get_ohlcv` (see `phi/data/unified_data.py` and `docs/ecosystem_integration.md`). Optional live hook: env `PHINANCE_ECOSYSTEM_OHLCV_HOOK=module:callable`.
- **Trading desk / options playbook**: Streamlit sidebar page **Trading desk**; options playbook JSON override via `PHINANCE_OPTIONS_PLAYBOOK`. Options backtests can attach `metrics_by_regime` when regime-aware blending + detector are enabled (`docs/options_regime_playbook.md`).
- **Optional services**: Ollama (for Plutus Bot tab) and Polygon.io (for L2 feed) are optional and the app gracefully degrades without them.
- **`python3.12-venv` system package**: Required to create the venv; install with `sudo apt-get install -y python3.12-venv` if not already present.

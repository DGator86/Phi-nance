# Phi-nance

[![Tests](https://github.com/DGator86/Phi-nance/actions/workflows/test.yml/badge.svg)](https://github.com/DGator86/Phi-nance/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/DGator86/Phi-nance/branch/main/graph/badge.svg)](https://codecov.io/gh/DGator86/Phi-nance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Phi-nance is a quantitative trading research platform for building, blending, and validating regime-aware strategies through a modular Python engine and Streamlit workbench.

## Key Features

- Regime-aware MFT engine and strategy research workflows.
- Multi-vendor data ingestion and cache-backed data spine.
- PostgreSQL options vendor support for intraday options research datasets.
- Indicator catalog + configurable signal blending (weighted, voting, regime-weighted).
- PhiAI optimization workflows for parameter tuning and walk-forward validation.
- Equity and options backtesting paths.
- Streamlit workbench and dashboards for interactive workflows.
- Centralized configuration, logging, custom exceptions, and validation helpers.
- **Ecosystem:** canonical OHLCV for Lumibot / TensorTrade / agent-cli adapters — see [docs/ecosystem_integration.md](docs/ecosystem_integration.md).
- **Options playbook:** regime × vol rows, transition map, and regime-tagged options metrics — see [docs/options_regime_playbook.md](docs/options_regime_playbook.md); Streamlit hub: sidebar **Trading desk**.

## Quick Start

```bash
# 1) Clone and enter the repo
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance

# 2) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Optional: JupyterLab — Phi-nance path bootstrap + regime demos
pip install -r requirements-jupyter.txt
python -m ipykernel install --user --name phinance --display-name "Python (Phi-nance)"
jupyter lab
# Open notebooks/00_getting_started.ipynb (first cell runs notebook_setup.py)
# Easy-mode helpers without Streamlit: notebooks/10_easy_strategy_lab.ipynb

# 4) Configure environment
cp .env.example .env

# 5) Build/refresh local data spine
python scripts/setup_data_spine.py --tickers SPY QQQ --years 2

# 6) Run the modular Streamlit app (easy mode by default)
streamlit run app_streamlit/main.py --server.headless true
```

**Windows:** from repo root, `.\run_local.ps1` uses the venv and binds `127.0.0.1` (auto-picks the next free port if 8501 is busy).

Useful commands:

```bash
# Streamlit (easy mode: overview / auto backtest / ticker)
streamlit run app_streamlit/main.py --server.headless true

# Full expert workbench (trading desk, all tuning)
streamlit run app_streamlit/expert_workbench.py --server.headless true

# Engine health validation
python scripts/engine_health.py

# Quick import smoke test (venv + paths)
python scripts/quick_import_check.py

# CLI backtest
python scripts/run_backtest.py --symbol SPY --start 2020-01-01 --end 2024-12-31 --capital 100000

# Optional: Dash wallboard (SPY daily, 60s refresh) — separate from Streamlit
pip install -r requirements-dash.txt
python app_dash/app.py
```

## Project Layout

```text
Phi-nance/
├── app_streamlit/         # Streamlit UI (main=easy mode, expert_workbench, pages)
├── phi/                   # Core runtime package (config, logging, data, indicators, options)
├── phinance/              # Research/optimization and agent framework package
├── regime_engine/         # MFT regime engine and feature pipeline components
├── strategies/            # Strategy implementations used by backtest adapters
├── legacy/                # Deprecated Streamlit/Lumibot apps (do not extend)
├── scripts/               # Operational CLI scripts (fetching, backtests, setup)
├── tests/                 # Unit and integration-style tests
├── docs/                  # End-user and contributor documentation
├── run_local.ps1          # Windows: venv + Streamlit on localhost
└── start.sh               # Linux/macOS: venv + Streamlit
```

Deeper map: [`docs/architecture_layout.md`](docs/architecture_layout.md).

## Configuration

Runtime configuration is environment-driven (see `.env.example` and `docs/configuration.md`).

Common variables:

- `DATA_CACHE_DIR`: cache root for fetched datasets.
- `RUNS_DIR`: run artifacts and result bundles.
- `LOGS_DIR`: log output directory.
- `LOG_LEVEL`: logging verbosity (`DEBUG`, `INFO`, ...).
- `DEBUG`: toggles expanded debug details in UI error panels.
- `PHIAI_DEFAULT_N_TRIALS`, `PHIAI_PARALLEL_JOBS`, `PHIAI_WALK_FORWARD_WINDOWS`: PhiAI defaults.

## Usage Examples

```python
from phi.data.cache import fetch_and_cache

df = fetch_and_cache(
    vendor="yfinance",
    symbol="SPY",
    timeframe="1D",
    start="2022-01-01",
    end="2024-01-01",
)
```

```python
import pandas as pd
from phi.blending.blender import blend_signals

signals = pd.DataFrame({"RSI": rsi_signal, "MACD": macd_signal})
combined = blend_signals(
    signals,
    method="weighted_sum",
    weights={"RSI": 0.6, "MACD": 0.4},
)
```

```python
from phi.phiai.auto_tune import run_phiai_optimization

optimized, explanation = run_phiai_optimization(
    ohlcv,
    indicators={
        "RSI": {"enabled": True, "auto_tune": True, "params": {}},
        "MACD": {"enabled": True, "auto_tune": True, "params": {}},
    },
    max_iter_per_indicator=20,
)
```

```python
from phi.options.data_adapter import fetch_options_data

options_df = fetch_options_data(
    symbol="SPY",
    start="2022-01-10",
    end="2022-01-15",
)
```

See [`docs/postgres-options-vendor.md`](docs/postgres-options-vendor.md) for setup details.

## Documentation

- Repository layout: [`docs/architecture_layout.md`](docs/architecture_layout.md)
- Branching / workflow: [`docs/DEV_WORKFLOW.md`](docs/DEV_WORKFLOW.md)
- Architecture: [`docs/Architecture.md`](docs/Architecture.md)
- Contributor guide: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)
- Easy mode (Streamlit): [`docs/easy_mode.md`](docs/easy_mode.md)
- UI roadmap (visual backtest, future Dash, etc.): [`docs/ui_roadmap.md`](docs/ui_roadmap.md)
- Full docs index: [`docs/quickstart.md`](docs/quickstart.md)
- External options data landscape: [`docs/external-options-data-landscape.md`](docs/external-options-data-landscape.md)

## Advanced Usage

- Auto-training pipeline: [`docs/auto_training.md`](docs/auto_training.md)

## Contributing

Please follow the standards and workflow in [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md), including logging, validation, exception usage, typing, testing, and docs updates for user-facing behavior changes.

Optional [pre-commit](https://pre-commit.com/) hooks are configured in `.pre-commit-config.yaml` (`pip install pre-commit && pre-commit install`).

## Troubleshooting

- **Windows: use the venv interpreter** — Running `python -m streamlit ...` may use **AppData** Python 3.13 (wrong deps, slow protobuf import, or crashes). Prefer **`.\run_local.ps1`** or **`.\venv\Scripts\python.exe -m streamlit run ...`** from the repo root.
- **PowerShell 5.x** — `&&` is not valid; chain with **`;`** (e.g. `cd path; git pull origin MAIN`).
- **Port 8501 already in use** — [`run_local.ps1`](run_local.ps1) scans upward for a free port. To free 8501: `Get-NetTCPConnection -LocalPort 8501 -State Listen | Select-Object -ExpandProperty OwningProcess -Unique` then **`Stop-Process -Id <that_pid> -Force`** (use the real PID, not an example).
- **`127.0.0.1 refused to connect` / Streamlit won’t open** — The server is not running or the port is taken. Keep the terminal open, use [`run_local.ps1`](run_local.ps1) on Windows (it picks the next free port after 8501), or see [`docs/easy_mode.md`](docs/easy_mode.md).
- **Wrong or missing packages** — Activate the project **venv** and reinstall: `pip install -r requirements.txt`. Quick check: `python scripts/quick_import_check.py`. Full engine smoke test: `python scripts/engine_health.py`.
- **Doc links 404 on Linux/macOS** — Use **`docs/Architecture.md`** and **`docs/CONTRIBUTING.md`** (capital **A** / **C**). Duplicate lowercase paths were removed; see git history on `MAIN` if you see missing files after an old pull.
- **Backtest / easy-mode errors** — `git pull origin MAIN` and confirm you run the repo’s `phi` (not an older global install): `python -c "import phi.backtest.direct as d; print(d.__file__)"`.

## License

MIT. See [`LICENSE`](LICENSE).

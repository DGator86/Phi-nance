# Phi-nance

[![Tests](https://github.com/DGator86/Phi-nance/actions/workflows/test.yml/badge.svg)](https://github.com/DGator86/Phi-nance/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/DGator86/Phi-nance/branch/main/graph/badge.svg)](https://codecov.io/gh/DGator86/Phi-nance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Phi-nance is a quantitative trading research platform for building, blending, and validating regime-aware strategies through a modular Python engine and Streamlit workbench.

## Key Features

- Regime-aware MFT engine and strategy research workflows.
- Multi-vendor data ingestion and cache-backed data spine.
- Indicator catalog + configurable signal blending (weighted, voting, regime-weighted).
- PhiAI optimization workflows for parameter tuning and walk-forward validation.
- Equity and options backtesting paths.
- Streamlit workbench and dashboards for interactive workflows.
- Centralized configuration, logging, custom exceptions, and validation helpers.

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

# 4) Configure environment
cp .env.example .env

# 5) Build/refresh local data spine
python scripts/setup_data_spine.py --tickers SPY QQQ --years 2

# 6) Run the modular Streamlit app
streamlit run app_streamlit/main.py --server.headless true
```

Alternative entry points:

```bash
# Legacy dashboard
streamlit run dashboard.py --server.headless true

# Engine health validation
python engine_health.py

# CLI backtest
python run_backtest.py --strategy rsi --start 2020-01-01 --end 2024-12-31 --budget 100000
```

## Project Layout

```text
Phi-nance/
├── app_streamlit/         # Streamlit UI modules and pages
├── phi/                   # Core runtime package (config, logging, data, indicators, options)
├── phinance/              # Research/optimization and agent framework package
├── regime_engine/         # MFT regime engine and feature pipeline components
├── strategies/            # Strategy implementations used by backtest adapters
├── scripts/               # Operational CLI scripts (fetching, backtests, setup)
├── tests/                 # Unit and integration-style tests
├── docs/                  # End-user and contributor documentation
├── Architecture.md        # System architecture and module map
└── CONTRIBUTING.md        # Contributor workflow and standards
```

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

## Documentation

- Architecture: [`Architecture.md`](Architecture.md)
- Contributor guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Full docs index: [`docs/quickstart.md`](docs/quickstart.md)
- External options data landscape: [`docs/external-options-data-landscape.md`](docs/external-options-data-landscape.md)

## Contributing

Please follow the standards and workflow in [`CONTRIBUTING.md`](CONTRIBUTING.md), including logging, validation, exception usage, typing, testing, and docs updates for user-facing behavior changes.

## License

MIT. See [`LICENSE`](LICENSE).

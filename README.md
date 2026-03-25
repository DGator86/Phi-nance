# Phi-nance (QuantConnect-oriented)

[![Tests](https://github.com/DGator86/Phi-nance/actions/workflows/test.yml/badge.svg)](https://github.com/DGator86/Phi-nance/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python **research library** for regime-aware signals, options analytics, and backtests, plus a **QuantConnect / Lean** bridge. There is **no web UI** in this repository—run research via scripts, notebooks, or the optional headless HTTP exporter; run **live and paper trading on QuantConnect** (or consume exported bundles there).

## Quick start

```bash
git clone https://github.com/DGator86/Phi-nance.git
cd Phi-nance
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, for pytest
cp .env.example .env

# Headless export API (bundles for QC custom data)
./start.sh
# → http://0.0.0.0:8080/health
```

**Windows:** `.\run_local.ps1` (binds `127.0.0.1:8080`).

**Export a bundle for QuantConnect (CSV + manifest):**

```bash
python scripts/export_quantconnect_bundle.py \
  --symbol SPY --start 2022-01-01 --end 2024-12-31 \
  --out-dir ./exports/qc_SPY \
  --include-signal-card
```

**Lean template:** copy `quantconnect/main.py` into your QC project and point `GetSource` at the uploaded `ohlcv.csv`. Full steps: [docs/quantconnect_deployment.md](docs/quantconnect_deployment.md).

**CLI backtest (local research):**

```bash
python scripts/run_backtest.py --symbol SPY --start 2020-01-01 --end 2024-12-31 --capital 100000
```

**Engine smoke test:** `python scripts/engine_health.py`  
**Import check:** `python scripts/quick_import_check.py`

## Docker

```bash
docker compose up --build -d
# API on port 8080; optional worker profile: docker compose --profile worker up -d
```

## Project layout

```text
Phi-nance/
├── quantconnect/       # Lean Python algorithm template (paste into QC cloud)
├── phi/                # Core package: data, regime, indicators, options, backtest
├── phinance/           # Research, blending, experiments, RL hooks
├── regime_engine/      # MFT regime components
├── strategies/         # Strategy implementations for adapters
├── scripts/            # CLIs (export, backtest, data spine, health)
├── tests/
├── docs/
├── start.sh            # Linux/macOS: uvicorn export API
└── run_local.ps1       # Windows: same on 8080
```

Agent / Jupyter notes: [docs/AGENTS.md](docs/AGENTS.md). Ecosystem (Lumibot, TensorTrade, etc.): [docs/ecosystem_integration.md](docs/ecosystem_integration.md).

## Configuration

Environment-driven (`.env.example`, `docs/configuration.md`). Common vars: `DATA_CACHE_DIR`, `UNUSUAL_WHALES_API_KEY`, `PHINANCE_QC_EXPORT_DIR`, `PHINANCE_LOG_JSON`.

## Documentation

- **QuantConnect:** [docs/quantconnect_deployment.md](docs/quantconnect_deployment.md)
- Layout / workflow: [docs/architecture_layout.md](docs/architecture_layout.md), [docs/DEV_WORKFLOW.md](docs/DEV_WORKFLOW.md)
- Options playbook: [docs/options_regime_playbook.md](docs/options_regime_playbook.md)

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Many older docs still mention Streamlit; treat them as historical unless updated.

## License

MIT. See [LICENSE](LICENSE).

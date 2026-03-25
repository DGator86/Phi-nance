# AGENTS.md

## Overview

Phi-nance is organized for **QuantConnect-first** workflows: a **Python research library** (`phi/`, `phinance/`, `regime_engine/`) used locally or in CI, plus a **Lean algorithm template** under `quantconnect/` to copy into the [QuantConnect](https://www.quantconnect.com/) cloud IDE. There is **no Streamlit or bundled web UI** in this branch.

**Layout:** `docs/architecture_layout.md` (may still mention removed UI paths — prefer this file and `README.md`).

### Running headless services

```bash
source venv/bin/activate
./start.sh
# or: uvicorn phi.api.qc_export:app --host 0.0.0.0 --port 8080
```

Default HTTP port **8080**. Set `PHINANCE_QC_EXPORT_DIR` for bundle output. `PHINANCE_LOG_JSON=1` enables JSON logs.

**Windows:** `.\run_local.ps1`

### QuantConnect

1. Read [quantconnect_deployment.md](quantconnect_deployment.md).
2. Copy `quantconnect/main.py` into a QC Python project; upload `ohlcv.csv` from `scripts/export_quantconnect_bundle.py`.
3. Live trading and backtests run **on QuantConnect**, not via a local GUI.

### JupyterLab (optional research)

```bash
pip install -r requirements-jupyter.txt
jupyter lab
```

Use `notebooks/00_getting_started.ipynb` and `notebook_setup.py` as before.

### Regime training (YAML + optional MLflow)

```bash
cp configs/ml/regime_train.example.yaml configs/ml/regime_train.yaml
# edit dates/symbol, then:
phi-regime-train --config configs/ml/regime_train.yaml
# or: export PHINANCE_REGIME_TRAIN_CONFIG=... && phi-regime-train
```

See [ml_components.md](ml_components.md).

### Validating the engine

```bash
python scripts/engine_health.py
```

### Key caveats

- **`IS_BACKTESTING` env var**: Set before importing lumibot where applicable (e.g. `notebook_setup.py`).
- **Tests**: `pytest` under `tests/`; see `.github/workflows/test.yml`.
- **`.env`**: Copy `.env.example` to `.env` for vendor keys (Unusual Whales, Alpaca, etc.).
- **Ecosystem OHLCV**: `phi.data` unified fetch — [ecosystem_integration.md](ecosystem_integration.md).
- **Options playbook**: JSON via `PHINANCE_OPTIONS_PLAYBOOK`; see [options_regime_playbook.md](options_regime_playbook.md).
- **Duplicate-looking strategy trees**: Prefer `phinance.strategies` for vectorized research; top-level `strategies/` is the Lumibot-oriented legacy bundle. Discrete regime → playbook: `phi.regime.strategy_mapping` (also on `phi.options`). Continuous tensor → force-field ranking: `phi.force_field`.

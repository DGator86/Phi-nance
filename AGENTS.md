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

### Validating the engine

```bash
source venv/bin/activate
python engine_health.py
```

Runs the full MFT pipeline on synthetic OHLCV data (no API key needed). Exit code 0 = all 6 components pass.

### Key caveats

- **`IS_BACKTESTING` env var**: The dashboard auto-sets `IS_BACKTESTING=True` before importing lumibot. Without this, lumibot's `credentials.py` crashes trying to instantiate live brokers. This is already handled in `dashboard.py` line 34.
- **No automated test suite**: There are no `test_*.py` or `*_test.py` files. The closest validation is `engine_health.py`. pytest is installed (as a lumibot dependency) but there are no test files to run.
- **`.env` file**: Copy `.env.example` to `.env`. The default `AV_API_KEY` in `.env.example` is a free-tier Alpha Vantage key (rate-limited to 5 req/min). Backtests and data fetching require this key.
- **Optional services**: Ollama (for Plutus Bot tab) and Polygon.io (for L2 feed) are optional and the app gracefully degrades without them.
- **`python3.12-venv` system package**: Required to create the venv; install with `sudo apt-get install -y python3.12-venv` if not already present.

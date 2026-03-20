# Quickstart

## 1. Install and configure

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp .env.example .env
```

## 2. Build data spine

```bash
python scripts/setup_data_spine.py --tickers SPY QQQ --years 2
```

## 3. Validate engine health

```bash
python scripts/engine_health.py
```

## 4. Launch Streamlit workbench

```bash
streamlit run app_streamlit/main.py --server.headless true
```

## 5. Run a CLI backtest

```bash
python scripts/run_backtest.py --symbol SPY --start 2020-01-01 --end 2024-12-31 --capital 100000
```

## 6. Inspect outputs

- Run artifacts in `${RUNS_DIR}`
- Logs in `${LOGS_DIR}`
- Cached data in `${DATA_CACHE_DIR}`

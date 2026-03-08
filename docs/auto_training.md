# Auto-Training Pipeline for PhiAI

## Overview

The auto-training pipeline retrains PhiAI indicator parameters on a regular cadence using fresh OHLCV data. It is intended to keep optimization outputs aligned with changing market regimes, then persist best-parameter artifacts for reuse in dashboards and custom workflows.

Use `scripts/auto_train.py` to:

- Fetch and cache historical data per ticker.
- Run PhiAI optimization (Optuna + walk-forward scoring).
- Save best parameter bundles to an output directory.

## Installation

No additional packages are required beyond the repository defaults:

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Configuration

There are no script-specific environment variables.

However, provider credentials may be required depending on the selected `--vendor`:

- `yfinance`: usually no key required.
- `alphavantage`: API key/rate limits may apply depending on your setup.

You can also configure shared runtime paths and log settings via `.env` (for example `DATA_CACHE_DIR`, `RUNS_DIR`, `LOG_LEVEL`).

## Running Manually

Basic usage:

```bash
python scripts/auto_train.py --tickers SPY QQQ
```

Higher-effort run with more trials and parallel workers:

```bash
python scripts/auto_train.py \
  --tickers SPY QQQ IWM \
  --years 3 \
  --timeframe 1D \
  --n-trials 100 \
  --windows 4 \
  --parallel 4 \
  --metric sharpe \
  --output-dir ./runs/best_params \
  --vendor yfinance \
  --force-refresh
```

### Options

- `--tickers` (required): space-separated symbols.
- `--timeframe` (default: `1D`): bar interval.
- `--years` (default: `3`): lookback length.
- `--end-date` (default: today): `YYYY-MM-DD`.
- `--n-trials` (default: `50`): Optuna trials.
- `--windows` (default: `3`): walk-forward windows.
- `--parallel` (default: `1`): parallel optimization jobs.
- `--metric` (default: `sharpe`): optimization target.
- `--output-dir` (default: `./runs/best_params`): parameter artifact path.
- `--vendor` (default: `yfinance`): data source.
- `--force-refresh`: bypass cache.
- `--verbose`: enable DEBUG logging for the script logger.

## Automated Scheduling

### Cron (Linux/macOS)

Example: run daily at 01:15.

```cron
15 1 * * * cd /path/to/Phi-nance && /path/to/Phi-nance/venv/bin/python scripts/auto_train.py --tickers SPY QQQ --years 3 --n-trials 100 --parallel 4 >> /path/to/Phi-nance/logs/auto_train.log 2>&1
```

Notes:

- Use absolute paths in cron.
- Redirect stdout/stderr for observability.

### Task Scheduler (Windows)

1. Open **Task Scheduler** and select **Create Task**.
2. In **General**, choose a task name like `PhiAI Auto Training`.
3. In **Triggers**, create a schedule (e.g., daily 1:15 AM).
4. In **Actions**, set:
   - **Program/script**: path to `python.exe` in your venv.
   - **Add arguments**: `scripts/auto_train.py --tickers SPY QQQ --years 3 --n-trials 100 --parallel 4`.
   - **Start in**: repository root directory.
5. In **Conditions/Settings**, adjust retry behavior as needed.
6. Save and run once manually to confirm logs/artifacts.

### GitHub Actions

Sample workflow (`.github/workflows/auto-train.yml`):

```yaml
name: Auto Train PhiAI

on:
  schedule:
    - cron: "15 1 * * *"
  workflow_dispatch:

jobs:
  auto-train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run auto training
        env:
          AV_API_KEY: ${{ secrets.AV_API_KEY }}
        run: |
          python scripts/auto_train.py --tickers SPY QQQ --years 3 --n-trials 100 --parallel 4
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: phiai-best-params
          path: runs/best_params
```

Set any needed provider keys in repository **Settings → Secrets and variables → Actions**.

## Using the Results

Saved artifacts are JSON parameter bundles keyed by dataset id. You can load them in custom scripts:

```python
from phi.phiai.auto_tune import load_best_params

payload = load_best_params("SPY_1D_2022-01-01_2024-12-31")
if payload:
    print(payload["best_params"])
```

In Streamlit/custom flows, load params and merge them into indicator configuration before running signals/backtests.

## Troubleshooting

- **No data returned for ticker**
  - Validate symbol and vendor support.
  - Try `--force-refresh`.
  - Check network/API limits.
- **No optimization improvement**
  - Increase `--n-trials` and/or `--windows`.
  - Revisit indicator set and parameter grids.
- **Frequent vendor failures**
  - Verify API keys and quotas.
  - Add retries at scheduler level.
- **Output artifacts missing**
  - Confirm write permission for `--output-dir`.
  - Run with `--verbose` and inspect logs.

## Extending

To change what gets tuned, update `DEFAULT_INDICATORS` in `scripts/auto_train.py`.

Current default set:

- RSI
- MACD
- BBands

To add an indicator:

1. Ensure it is supported in your indicator registry/evaluation path.
2. Add it to `DEFAULT_INDICATORS` with `enabled=True` and `auto_tune=True`.
3. Define/verify parameter grids in `phi/phiai/auto_tune.py`.
4. Re-run auto-training and inspect output artifacts/explanations.

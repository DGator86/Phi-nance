# QuantConnect deployment — Phi-nance bridge

Phi-nance runs **outside** the QuantConnect (Lean) cloud. Deployment means: produce **reproducible artifacts** here, then **ingest** them in a Lean algorithm as custom data or Object Store files, or mirror the same logic with Lean indicators inside `QCAlgorithm`.

## 1. Export a bundle (CLI)

Uses canonical **Unusual Whales → yfinance** OHLCV (`phi.data.fetch_ohlcv_uw_then_yf`).

```bash
# From repo root; set UNUSUAL_WHALES_API_KEY in .env if you want UW first
python scripts/export_quantconnect_bundle.py \
  --symbol SPY --start 2022-01-01 --end 2024-12-31 \
  --timeframe 1D \
  --out-dir ./exports/qc_SPY_1D

# Optional: attach options signal card JSON (phi.options signal generator)
python scripts/export_quantconnect_bundle.py \
  --symbol SPY --start 2020-01-01 --end 2024-12-31 \
  --out-dir ./exports/qc_SPY_with_card \
  --include-signal-card
```

Output:

| File | Purpose |
|------|---------|
| `ohlcv.csv` | `time,open,high,low,close,volume` — Lean-friendly daily bars |
| `manifest.json` | Symbol, range, vendor used, schema version, QC doc links |
| `signal_card.json` | Optional; regime / playbook / MTF summary for research or manual rules |

Zip the folder and upload to the QC project **Data** tree or **Object Store**, then point a custom data class at that path (see Lean template in `quantconnect/main.py`).

## 2. Headless HTTP service (Docker / local)

FastAPI and uvicorn are **default dependencies** (`pip install -r requirements.txt` or `pip install -e .`).

```bash
export PHINANCE_QC_EXPORT_DIR=/tmp/qc_exports
uvicorn phi.api.qc_export:app --host 0.0.0.0 --port 8080
```

Or Docker Compose (service `api`):

```bash
docker compose up --build -d
# POST http://localhost:8080/export/bundle?symbol=SPY&start=2022-01-01&end=2024-12-31
```

Set `PHINANCE_LOG_JSON=1` for one JSON object per log line (useful behind Docker log drivers).

## 3. Lean algorithm (in QuantConnect cloud)

Copy `quantconnect/main.py` into a new Python project in the QC web IDE. Adjust:

- `SubscriptionDataSource` path to match where you placed `ohlcv.csv`.
- Symbol / resolution to match your bundle (`Resolution.Daily` for `1D` exports).

Full custom data patterns: [Custom securities](https://www.quantconnect.com/docs/v2/writing-algorithms/importing-data/streaming-data/custom-securities/key-concepts).

## 4. What stays in Phi-nance vs QC

| Concern | Phi-nance | QuantConnect |
|---------|-----------|--------------|
| Vendor OHLCV + cache | Yes | Use QC data or your CSV |
| Regime / options signal card | Yes (export JSON) | Reimplement or read JSON in research |
| Order execution / portfolio | Scripts / Alpaca where configured | QC brokerage models / live |

## 5. Tests

```bash
pytest tests/test_quantconnect_export.py -q
```

## 6. ML / regime labels on QC

For trained Phi-nance regime models, prefer exporting **per-bar regime CSV** for Lean custom data — see [quantconnect_ml_inference.md](quantconnect_ml_inference.md) and [ml_components.md](ml_components.md).


## 7. Troubleshooting local `lean backtest` (Windows + Docker)

If you run Lean CLI locally (instead of the QC cloud IDE), two common errors are:

- `No module named 'pkg_resources'`
- `No module named 'phi'`

### A) `pkg_resources` missing

`pkg_resources` is provided by `setuptools`. Lean CLI still expects it.

```powershell
# inside the venv you pass to --python-venv
python -m pip install --upgrade "setuptools<70"
python -c "import pkg_resources; print(pkg_resources.__file__)"
```

### B) Venv path not found inside Lean container

When Dockerized Lean starts, it runs in Linux. A Windows path like
`C:\Users\...\venv` is not valid in-container and can show up as `/C:\Users\...\venv`.

Use a path visible *inside* the Lean container:

- WSL example: `/mnt/c/Users/<you>/Phi-nance/venv`
- Or place your algorithm + dependencies directly in the Lean project folder and avoid external host paths.

### C) `No module named 'phi'`

Lean only imports modules available to the runtime used by the backtest.

- If your algorithm imports `phi.*`, install Phi-nance into the same interpreter Lean uses.
- Or keep Lean algorithms self-contained (recommended for QC deployment) and only ingest exported artifacts (`ohlcv.csv`, `signal_card.json`).

In practice, the most reliable deployment path for this repo is still: export bundle in Phi-nance → run strategy in QuantConnect using `quantconnect/main.py` pattern.


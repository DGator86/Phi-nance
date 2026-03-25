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

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

If you run Lean CLI locally (instead of the QC cloud IDE), common issues:

### A) `No module named 'pkg_resources'`

`pkg_resources` is provided by `setuptools`. Lean CLI still expects it.

```powershell
# inside the venv you pass to --python-venv
python -m pip install --upgrade "setuptools<70"
python -c "import pkg_resources; print(pkg_resources.__file__)"
```

### B) Venv path not found inside Lean container

`lean backtest` runs the engine in a **Linux container**. A Windows path like
`C:\Users\...\venv` is forwarded incorrectly and may appear as `/C:\Users\...\venv`
and **does not exist** in the container.

**Algorithms that only use the standard library plus `AlgorithmImports`** do not need
`--python-venv` — run:

```powershell
lean backtest PhiNanceExported
```

If you need `phi` (editable install) or other pip packages inside Lean, use a path
visible inside the container, not a bare `C:\...` host path:

- WSL-style path where the Lean workspace is mounted, e.g. `/mnt/c/Users/<you>/...`
- Or a **project-local** venv next to your Lean project (see below).

Recommended Windows pattern when the algorithm imports `phi`:

```powershell
# from your Lean project folder (same folder you pass to `lean backtest`)
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip "setuptools<70"
.\.venv\Scripts\python -m pip install -e "C:\Users\<you>\OneDrive - Penetron\Desktop\Phi-nance"
lean backtest PhiNanceExported --python-venv ".\.venv"
```

Using a project-local `.venv` avoids pointing Lean at a **different** host tree than
the one mounts into Docker (for example `Phi-nance` under `OneDrive\Desktop` vs
`C:\Users\<you>\Phi-nance`).

### C) Custom `LocalFile` CSV “missing” — file in the wrong folder

For **Lean CLI** backtests, QuantConnect’s local custom-data pattern joins
**`Globals.DataFolder`** with your filename. That folder is the **organization
workspace** `data/` directory (see root `lean.json` → `"data-folder": "data"`), i.e.
next to your project folders — **not** necessarily `<project>/data/`.

The template in `quantconnect/main.py` uses:

`os.path.join(str(Globals.DataFolder), "ohlcv.csv")`

Copy the export to:

```text
<LeanWorkspace>/data/ohlcv.csv
```

Example: `C:\Users\you\LeanWorkspace\data\ohlcv.csv` when the project is
`...\LeanWorkspace\PhiNanceExported`.

Use `scripts/sync_lean_phi_nance_export.ps1`, which writes to workspace `data/` (and
mirrors under `<project>/data/` for convenience). Putting CSV only at the project root
or only under `<project>/data/` without the workspace copy yields almost no custom
bars, **`Total Orders 0`**, and failed data requests in the monitor.

**QC cloud:** upload under the project Data tree per cloud docs; `Globals.DataFolder`
there matches the cloud data root.

### D) `No module named 'phi'`

Lean only imports modules available to the interpreter used by the backtest.

- Install Phi-nance into that environment (see B), **or**
- Keep Lean algorithms self-contained and only ingest exported artifacts (`ohlcv.csv`,
  `signal_card.json`) — recommended for QC cloud deployment.

### E) `⚠️ Could not load signal_card` / `signal_card.json`

Informational when optional JSON is missing or unreadable; Lean continues in basic mode.

- Re-export with `--include-signal-card`, then place `signal_card.json` where your
  algorithm expects it (often the Lean project root). Resolve paths from
  `Path(__file__).resolve().parent` so loading works when the project is mounted as
  `/LeanCLI` in the container.
- For OHLCV-only strategies, you can ignore the warning.

In practice, the most reliable path for this repo remains: export the bundle in
Phi-nance → run the strategy in QuantConnect using the `quantconnect/main.py` pattern.

## 8. Local Lean CLI — sync script (Windows)

The sync helper lives in **this repo** (`scripts/sync_lean_phi_nance_export.ps1`), not in
`LeanWorkspace`. From your Lean folder, call it with a **full path** to the script.

**Auto-find the newest export** (if you are unsure which folder under `exports/` has CSV):

```powershell
& "C:\Users\<you>\Phi-nance\scripts\sync_lean_phi_nance_export.ps1" `
  -PhiNanceRoot "C:\Users\<you>\Phi-nance" `
  -LeanProject "C:\Users\<you>\LeanWorkspace\PhiNanceExported"
```

Use **OneDrive** `PhiNanceRoot` if that is where you run the export CLI. If OneDrive has
no `exports\...\ohlcv.csv` yet, either export there or point `-PhiNanceRoot` at the clone
that already has `exports\` (for example `C:\Users\<you>\Phi-nance`).

**Explicit folder:**

```powershell
& "C:\Users\<you>\Phi-nance\scripts\sync_lean_phi_nance_export.ps1" `
  -ExportDir "C:\Users\<you>\Phi-nance\exports\qc_SPY_daily_uw" `
  -LeanProject "C:\Users\<you>\LeanWorkspace\PhiNanceExported"
```

Or `cd` to the Phi-nance repo root and use `.\scripts\sync_lean_phi_nance_export.ps1 ...`.

Then:

```powershell
cd C:\Users\<you>\LeanWorkspace
lean backtest PhiNanceExported
```

Do **not** pass `--python-venv` with a `C:\...` path when Docker runs the engine (see §7).

**After `git pull`,** refresh the algorithm file: copy `quantconnect/main.py` over
`LeanWorkspace\PhiNanceExported\main.py`. If the backtest log still shows
`Dates: Start: 01/01/2022`, your Lean copy is stale (current template starts 2023-01-03
and uses `Globals.DataFolder` for CSV).

The template algorithm is `PhiNanceBridgeAlgorithm` in `quantconnect/main.py` (optional
`signal_card.json` next to `main.py`; **`ohlcv.csv` in `<LeanWorkspace>/data/`** for CLI).

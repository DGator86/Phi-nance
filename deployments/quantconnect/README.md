# QuantConnect Lean template

Copy `main.py` into a **Python** algorithm in the [QuantConnect](https://www.quantconnect.com/) web IDE (or local Lean). It is a **starting point** for ingesting `ohlcv.csv` produced by `scripts/export_quantconnect_bundle.py`.

- Full workflow: [docs/quantconnect_deployment.md](../../docs/quantconnect_deployment.md)
- Adjust the `SubscriptionDataSource` path to match your uploaded file (Object Store URL or project-relative path under `Data/`).

This file is **not** executed in the Phi-nance repo CI; Lean provides `AlgorithmImports` at runtime on QC.

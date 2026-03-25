# Machine learning components — status and enacted tooling

This document aligns with the ML-focused SWOT / Start–Stop–Continue analysis and records what is **implemented in the repo** vs **still manual / future**.

## Regime detection

**Supported trainers** (see `phi.regime.train`): **HMM** (`hmmlearn`), **KMeans**, **GMM** (sklearn), plus **deep** LSTM/transformer (`phi.regime.train_deep`, optional PyTorch).

**Random Forest** as a first-class regime classifier is **not** wired into `train_regime_detector` today; tree ensembles appear elsewhere (e.g. strategies, PhiAI search). The roadmap can add `method: random_forest` with pseudo-labels if needed.

### Enacted

| Capability | Location |
|------------|----------|
| **YAML-driven training** | `configs/ml/regime_train.example.yaml`, `phi.regime.cli_train`, `phi-regime-train` console script |
| **Optional `save_path`** | `phi.regime.train.train_regime_detector(..., save_path=...)` |
| **Training manifest** (config + metrics) | `phi.regime.artifacts.write_regime_training_manifest` → `<model>.manifest.json` |
| **Regime-oriented metrics** (not just accuracy) | `phi.regime.evaluation`: normalized entropy, transition rate, mean run length, forward-return spread by regime |
| **Explainability (lightweight)** | `phi.regime.explain.feature_regime_correlation_proxy`; optional SHAP+RF surrogate if `shap` installed and `explain.shap_surrogate: true` in YAML |
| **Optional MLflow** | Enable in YAML under `mlflow.enabled`; logs params, metrics, artifacts |
| **Scheduled retrain hook** | Set `PHINANCE_REGIME_TRAIN_CONFIG` and run `phi-regime-train` from cron / CI |
| **QC-friendly regime CSV** | `scripts/export_regime_predictions_for_qc.py` → `time,regime` for custom data |
| **Bundle copy for QC folder** | `phi.regime.artifacts.export_regime_bundle_for_quantconnect` |

### Still recommended (not fully automated)

- **Promotion rules** after retrain (compare new vs production model on walk-forward Sharpe / stability before swap).
- **Model registry** discipline (MLflow Model Registry or tagged paths).
- **Random Forest / XGBoost/LightGBM** regime heads as explicit `method` values if product requires them.

## PhiAI optimization

Walk-forward and Optuna integration remain in `phi.phiai` / run config. **Documentation** for adding a custom objective or classifier is still thin; extend this file or `docs/experimentation.md` as you add examples.

## Dependencies

- **Core training** needs **scikit-learn** / **hmmlearn** (see `requirements.txt` / `[project.optional-dependencies] ml`).
- **MLflow**: optional (`pip install mlflow` or `experiment` extra).
- **SHAP**: optional for surrogate explain block.

## Quick commands

```bash
# Copy example config and edit paths/dates
cp configs/ml/regime_train.example.yaml configs/ml/regime_train.yaml
phi-regime-train --config configs/ml/regime_train.yaml

# Regime series for QuantConnect custom data
python scripts/export_regime_predictions_for_qc.py \
  --model runs/regime_models/your.pkl --symbol SPY \
  --start 2022-01-01 --end 2024-12-31 --out exports/regimes_spy.csv
```

See also [quantconnect_ml_inference.md](quantconnect_ml_inference.md).

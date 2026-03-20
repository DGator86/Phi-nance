# Regime-Aware Auto-Training

Use regime-aware optimization to tune indicator parameters, regime detector settings, and per-regime indicator boosts in one flow.

## Command

```bash
python scripts/auto_train.py --tickers SPY --regime-optimize --n-trials 100
```

## What gets optimized

- Indicator parameters (existing PhiAI Optuna flow).
- Regime detector type: `hmm`, `kmeans`, `gmm`.
- Detector hyperparameters:
  - `n_regimes` (2–5)
  - `feature_window` (10–50)
  - `covariance_type` for HMM (`full`, `diag`)
- Per-regime boost matrix with values in `[0.5, 2.0]` for each enabled indicator.

## Output JSON

Saved under `runs/best_params/<dataset_id>.json` and includes:

- `dataset_id`
- `metric`
- `best_value`
- `indicators`
- `regime_detector`
- `regime_boosts`

## Loading and using the config

- Streamlit handlers can load a saved config payload for UI state hydration.
- Direct backtests accept `regime_detector_params` and `regime_boosts`; detector is created and fit from config.

## Caution

Regime-aware optimization adds many parameters and can overfit quickly. Always validate with strict out-of-sample data and/or separate walk-forward periods.

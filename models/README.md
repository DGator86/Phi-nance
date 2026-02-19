# Phi-nance — Trained Model Files

This directory stores fitted ML model files used by the ML strategies.

## Files

| File | Strategy | Description |
|------|----------|-------------|
| `classifier_rf.pkl`  | `MLClassifierStrategy` (model_type=random_forest)   | sklearn Random Forest |
| `classifier_gb.pkl`  | `MLClassifierStrategy` (model_type=gradient_boosting)| sklearn Gradient Boosting |
| `classifier_lr.pkl`  | `MLClassifierStrategy` (model_type=logistic)         | sklearn Logistic Regression |
| `classifier_lgb.txt` | `LightGBMStrategy` + `EnsembleMLStrategy`           | LightGBM binary classifier |
| `rl_agent_ppo.zip`   | `RLStrategy`                                        | Stable-Baselines3 PPO agent |

## How to Generate

```bash
# 1. Generate historical_regime_features.csv from your OHLCV data
#    (FeatureEngine output + direction label column)

# 2. Train sklearn + LightGBM models
python train_ml_classifier.py

# 3. (Optional) Train RL agent
python train_rl_agent.py
```

## Notes

- Model files are **not committed to git** (see `.gitignore`).
- Strategies run safely without model files — they produce `NEUTRAL`
  predictions and no trades until you train and place models here.

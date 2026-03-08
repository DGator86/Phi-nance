# Regime Detection

Phi now includes a trainable regime detection subsystem under `phi.regime`.

## Why this matters

Regimes (bull, bear, range, high-volatility) are useful context for dynamic weighting and strategy behavior. The detector outputs a per-bar string label that can be fed into blending/backtesting.

## Available detectors

- **HMM (`HMMRegimeDetector`)**
  - Uses `hmmlearn.hmm.GaussianHMM`
  - Learns latent sequential states
  - Key parameter: `n_states`

- **Clustering (`ClusteringRegimeDetector`)**
  - Uses `sklearn.cluster.KMeans` (or GMM mode in class)
  - Assigns each bar to a feature cluster
  - Key parameter: `n_clusters`

## Features used

`extract_features()` computes:

- returns
- log returns
- rolling volatility
- ATR ratio
- volume change

## Programmatic usage

```python
from phi.regime.train import train_regime_detector

detector, model_path = train_regime_detector(
    ohlcv=data,
    method="hmm",      # or "kmeans"
    n_regimes=3,
    window=20,
    save=True,
)

regime_series = detector.predict(data)
```

Models are saved to `settings.REGIME_MODELS_DIR` (default: `runs/regime_models`) with a JSON metadata sidecar.

## Streamlit flow

In the **Regime Detection** sidebar expander:

1. Enable regime-aware blending.
2. Choose method (HMM or Clustering/KMeans).
3. Set number of regimes and feature window.
4. Click **Train on selected range**.
5. Run backtest.

When enabled, the trained detector predicts regimes for backtest data and the results page shows a regime-overlay chart.

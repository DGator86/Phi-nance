# Deep Learning Regime Detection

This guide adds supervised deep-learning regime detectors (LSTM and Transformer) to `phi.regime`.

## Why deep models?

Traditional HMM/clustering methods are fast and robust, but they can miss nonlinear temporal structure. LSTM/attention models can:

- Learn longer contextual dependencies across bars.
- Combine many engineered features (returns, volatility, order flow proxies, entropy) into one classifier.
- Distill pseudo-labels from legacy detectors into a smoother classifier.

## Training data and labels

Current implementation is supervised. You must provide labels directly, or generate pseudo-labels from clustering/HMM first.

Suggested workflow:
1. Run `train_regime_detector` (HMM/KMeans) to get baseline labels.
2. Export those labels to CSV.
3. Train `DeepRegimeDetector` on historical OHLCV + labels.

## CLI usage

```bash
python -m phi.regime.train_deep \
  --ohlcv data/spy_1d.csv \
  --labels data/spy_regimes.csv \
  --model-type lstm \
  --seq-length 20 \
  --hidden-size 64 \
  --num-layers 2 \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --output models/regime/deep_lstm_spy.pkl
```

If `--labels` is omitted, pseudo-labels are generated with KMeans.

## Streamlit integration

In the **Regime-Aware Blending** panel:

- Choose **Deep Learning (LSTM)** or **Deep Learning (Transformer)**.
- Set sequence length / hidden size / layers / epochs / batch size / LR.
- Upload a pre-trained `.pkl` detector artifact.

The UI is designed to **load pre-trained deep models**. On-the-fly training in Streamlit is intentionally not supported due to latency.

## Backtest and live behavior

`DeepRegimeDetector.predict` returns a full-length regime series aligned to OHLCV index, with warmup rows as `NaN` for the first `seq_length - 1` rows.

This keeps behavior consistent with existing regime-aware blending logic and avoids incorrect backfilling during warmup.

## Overfitting guidance

- Start with small networks (`hidden_size=64`, `num_layers=2`).
- Use validation split and early stopping (`patience`).
- Prefer pseudo-label sets from multiple market regimes and symbols.
- Retrain periodically instead of overfitting a single period.

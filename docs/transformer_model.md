# Transformer Market Representation Model

## Overview

Phase 3 introduces a lightweight transformer encoder that learns temporal market structure and exposes latent embeddings to RL agents.

## Data Pipeline

The training dataset (`phinance/ml/data.py`) builds sliding windows from OHLCV data and computes engineered features:

- Returns and log-returns
- 20-day realized volatility
- RSI(14)
- MACD + signal
- Bollinger %b
- Volume z-score

`MarketSequenceDataset` normalizes features with `FeatureScaler` and prepares next-day return labels.

## Model Architecture

`phinance/ml/transformer.py` contains:

- Linear input projection to `d_model`
- Learned positional embeddings
- `nn.TransformerEncoder` stack
- Regression head for next-day return prediction

The final hidden state (`pooling=last`) is used as the embedding vector.

## Training

Run:

```bash
python -m phinance.ml.train_transformer --config configs/transformer_config.yaml
```

The trainer supports:

- Train/validation split
- AdamW optimization
- Early stopping
- Checkpoint persistence with model config and scaler metadata

Default checkpoint path:

`phinance/ml/checkpoints/transformer_latest.pt`

## Agent Integration

Execution, Strategy R&D, and Risk Monitor agents support two new options:

- `use_transformer_embeddings`
- `transformer_model_path`

When enabled, agent state vectors are augmented by transformer embeddings. Existing behavior remains unchanged when disabled.

## Inference and Caching

`TransformerFeatureExtractor`:

- Loads model/scaler metadata from checkpoint
- Accepts a market history DataFrame and emits embedding vectors
- Caches repeated windows with an LRU-like in-memory map
- Supports `extract_for_symbol(...)` for DataSourceManager-backed fetches

## Performance Notes

- Default architecture (`d_model=128`, 4 layers) is suitable for CPU inference under low-latency workloads.
- For stricter latency targets, reduce layers or d_model.
- Keep sequence length near 40-80 for a speed/quality balance.

## Fine-tuning and Versioning

- Periodic retraining can run from latest cached historical bars.
- Save new checkpoints under `phinance/ml/checkpoints/`.
- Metadata in the checkpoint supports update-manager driven model version tracking downstream.

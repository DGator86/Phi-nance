from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phinance.ml.data import MarketSequenceDataset
from phinance.ml.inference import TransformerFeatureExtractor
from phinance.ml.transformer import MarketTransformer, MarketTransformerConfig


def _frame(rows: int = 180) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100.0 + idx * 0.2 + np.sin(idx / 3.0)
    return pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 900_000 + idx * 50,
        }
    )


def test_transformer_forward_and_embed() -> None:
    cfg = MarketTransformerConfig(input_dim=12, d_model=32, num_layers=2, num_heads=4, max_seq_len=32)
    model = MarketTransformer(cfg)
    x = torch.randn(4, 20, 12)
    pred, contextual = model(x)
    emb = model.embed(x)
    assert pred.shape == (4,)
    assert contextual.shape == (4, 20, 32)
    assert emb.shape == (4, 32)


def test_checkpoint_load_and_extractor_embedding() -> None:
    ds = MarketSequenceDataset([_frame()], sequence_length=30, fit_scaler=True)
    cfg = MarketTransformerConfig(input_dim=len(ds.feature_columns), d_model=16, num_layers=1, num_heads=2, max_seq_len=30)
    model = MarketTransformer(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": cfg.to_dict(),
                "feature_columns": ds.feature_columns,
                "sequence_length": ds.sequence_length,
                "scaler": ds.scaler.to_dict(),
            },
            path,
        )

        extractor = TransformerFeatureExtractor(path)
        emb = extractor.extract_embedding(_frame(rows=80))
        assert emb.shape == (cfg.d_model,)

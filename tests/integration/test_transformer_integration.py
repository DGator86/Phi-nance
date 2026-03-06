from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from phinance.agents.execution import ExecutionAgent
from phinance.ml.data import MarketSequenceDataset
from phinance.ml.transformer import MarketTransformer, MarketTransformerConfig


def _frame(rows: int = 240) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100.0 + idx * 0.1 + np.sin(idx / 4.0)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1_100_000 + idx * 40,
        }
    )


def test_tiny_training_reduces_loss_and_execution_agent_uses_embeddings(tmp_path: Path) -> None:
    frame = _frame()
    ds = MarketSequenceDataset([frame], sequence_length=40, fit_scaler=True)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    cfg = MarketTransformerConfig(
        input_dim=len(ds.feature_columns),
        d_model=24,
        num_layers=1,
        num_heads=2,
        dim_feedforward=64,
        max_seq_len=40,
    )
    model = MarketTransformer(cfg)
    loss_fn = nn.MSELoss()
    opt = AdamW(model.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(3):
        running = 0.0
        count = 0
        for x, y in loader:
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            count += 1
        losses.append(running / max(count, 1))

    assert losses[-1] <= losses[0]

    ckpt = tmp_path / "transformer.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": cfg.to_dict(),
            "feature_columns": ds.feature_columns,
            "sequence_length": ds.sequence_length,
            "scaler": ds.scaler.to_dict(),
        },
        ckpt,
    )

    agent = ExecutionAgent(use_rl=False, use_transformer_embeddings=True, transformer_model_path=str(ckpt))
    state = agent._build_state({"qty": 100, "remaining_shares": 50, "horizon_steps": 5, "remaining_steps": 2}, frame.tail(60))
    assert state.shape[0] == 9 + cfg.d_model

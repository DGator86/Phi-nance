"""PyTorch transformer for market sequence prediction and embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class MarketTransformerConfig:
    """Hyperparameters for ``MarketTransformer``."""

    input_dim: int
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 256
    max_seq_len: int = 256

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MarketTransformer(nn.Module):
    """Transformer encoder with regression head and embedding output."""

    def __init__(self, config: MarketTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.regression_head = nn.Linear(config.d_model, 1)

        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prediction, embeddings)`` for an input batch ``[B, T, F]``."""
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq, features], got shape {tuple(x.shape)}")
        if x.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_seq_len={self.config.max_seq_len}")

        projected = self.input_projection(x)
        projected = projected + self.positional_embedding[:, : x.size(1), :]
        contextual = self.encoder(projected)
        contextual = self.norm(contextual)
        pooled = contextual[:, -1, :]
        prediction = self.regression_head(pooled).squeeze(-1)
        return prediction, contextual

    def embed(self, x: torch.Tensor, pooling: str = "last") -> torch.Tensor:
        """Return latent representation for each batch item."""
        _, contextual = self.forward(x)
        if pooling == "last":
            return contextual[:, -1, :]
        if pooling == "mean":
            return contextual.mean(dim=1)
        raise ValueError(f"Unsupported pooling strategy: {pooling}")

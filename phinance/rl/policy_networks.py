"""Policy network architectures for RL agents.

Supports MLP (default), recurrent LSTM, and transformer encoders so the same
training scripts can switch architectures via config.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _ensure_sequence(state: torch.Tensor, seq_len: int) -> torch.Tensor:
    if state.dim() == 2:
        state = state.unsqueeze(1).expand(-1, seq_len, -1)
    if state.dim() != 3:
        raise ValueError(f"Expected state rank 2 or 3, got shape={tuple(state.shape)}")
    return state


class _Backbone(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, architecture: str, sequence_length: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.architecture = architecture.lower()
        self.sequence_length = max(int(sequence_length), 1)

        if self.architecture == "mlp":
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
        elif self.architecture == "lstm":
            self.input_proj = nn.Linear(obs_dim, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        elif self.architecture == "transformer":
            self.input_proj = nn.Linear(obs_dim, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 2,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.positional = nn.Parameter(torch.zeros(1, self.sequence_length, hidden_size))
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'. Expected mlp, lstm, or transformer.")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.architecture == "mlp":
            if state.dim() == 3:
                state = state[:, -1, :]
            return self.net(state)

        seq = _ensure_sequence(state, self.sequence_length)
        x = self.input_proj(seq)
        if self.architecture == "lstm":
            out, _ = self.lstm(x)
            return out[:, -1, :]

        x = x + self.positional[:, : x.size(1), :]
        out = self.encoder(x)
        return out[:, -1, :]


class GaussianPolicy(nn.Module):
    """Continuous-action Gaussian policy for execution-like agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_size: int = 256,
        architecture: str = "mlp",
        sequence_length: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = _Backbone(obs_dim, hidden_size, architecture, sequence_length)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        mean = torch.sigmoid(self.mean_head(x))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class CategoricalPolicy(nn.Module):
    """Discrete-action policy for strategy/risk agents."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_size: int = 256,
        architecture: str = "mlp",
        sequence_length: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = _Backbone(obs_dim, hidden_size, architecture, sequence_length)
        self.logits_head = nn.Linear(hidden_size, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.backbone(state)
        return self.logits_head(x)


"""Training helpers for world models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from phinance.world.model import RSSMConfig, WorldModelRSSM


@dataclass
class TransitionBatch:
    """Mini-batch of transitions."""

    obs: Tensor
    actions: Tensor
    next_obs: Tensor
    rewards: Tensor
    dones: Tensor


class WorldModelTrainer:
    """Single-step world model trainer."""

    def __init__(
        self,
        model: WorldModelRSSM,
        learning_rate: float = 3e-4,
        reward_coef: float = 1.0,
        kl_coef: float = 0.1,
        done_coef: float = 0.1,
    ) -> None:
        self.model = model
        self.reward_coef = reward_coef
        self.kl_coef = kl_coef
        self.done_coef = done_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_loss(self, batch: TransitionBatch) -> Dict[str, Tensor]:
        prev_state = self.model.initial_state(batch.obs.shape[0], device=batch.obs.device)
        output = self.model.forward_step(batch.next_obs, batch.actions, prev_state)

        recon_loss = F.mse_loss(output["recon_obs"], batch.next_obs)
        reward_loss = F.mse_loss(output["reward"], batch.rewards)
        done_loss = F.binary_cross_entropy_with_logits(output["done_logit"], batch.dones)
        kl = self.model.kl_divergence(output["post_mean"], output["post_std"], output["prior_mean"], output["prior_std"]).mean()
        total = recon_loss + self.reward_coef * reward_loss + self.kl_coef * kl + self.done_coef * done_loss

        return {
            "loss": total,
            "recon_loss": recon_loss,
            "reward_loss": reward_loss,
            "kl_loss": kl,
            "done_loss": done_loss,
        }

    def train_step(self, batch: TransitionBatch) -> Dict[str, float]:
        self.model.train()
        losses = self.compute_loss(batch)
        self.optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return {k: float(v.detach().cpu().item()) for k, v in losses.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.model.config.__dict__,
            },
            path,
        )

    @staticmethod
    def load(path: Path, device: torch.device | None = None) -> WorldModelRSSM:
        payload = torch.load(path, map_location=device or "cpu")
        model = WorldModelRSSM(RSSMConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model


def build_transition_batch(
    obs: np.ndarray,
    actions: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    device: torch.device | None = None,
) -> TransitionBatch:
    """Convert numpy arrays to torch TransitionBatch."""
    device = device or torch.device("cpu")
    return TransitionBatch(
        obs=torch.as_tensor(obs, dtype=torch.float32, device=device),
        actions=torch.as_tensor(actions, dtype=torch.float32, device=device),
        next_obs=torch.as_tensor(next_obs, dtype=torch.float32, device=device),
        rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
    )

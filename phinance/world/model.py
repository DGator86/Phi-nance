"""Recurrent state-space world model (RSSM-inspired)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RSSMConfig:
    """Configuration for RSSM world model."""

    obs_dim: int
    action_dim: int
    det_dim: int = 128
    stoch_dim: int = 32
    hidden_dim: int = 128
    min_std: float = 0.1


@dataclass
class RSSMState:
    """Container for deterministic and stochastic latent state."""

    det: Tensor
    stoch: Tensor


class WorldModelRSSM(nn.Module):
    """Lightweight RSSM variant for market dynamics modeling."""

    def __init__(self, config: RSSMConfig) -> None:
        super().__init__()
        self.config = config

        self.obs_encoder = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(config.stoch_dim + config.action_dim, config.det_dim)
        self.prior_head = nn.Linear(config.det_dim, 2 * config.stoch_dim)
        self.posterior_head = nn.Linear(config.det_dim + config.hidden_dim, 2 * config.stoch_dim)

        self.decoder = nn.Sequential(
            nn.Linear(config.det_dim + config.stoch_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(config.det_dim + config.stoch_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(config.det_dim + config.stoch_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> RSSMState:
        device = device or next(self.parameters()).device
        det = torch.zeros(batch_size, self.config.det_dim, device=device)
        stoch = torch.zeros(batch_size, self.config.stoch_dim, device=device)
        return RSSMState(det=det, stoch=stoch)

    def _split_stats(self, stats: Tensor) -> Tuple[Tensor, Tensor]:
        mean, raw_std = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(raw_std) + self.config.min_std
        return mean, std

    def _sample(self, mean: Tensor, std: Tensor, sample: bool = True) -> Tensor:
        if sample and self.training:
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def transition(self, prev_state: RSSMState, action: Tensor) -> Dict[str, Tensor]:
        gru_input = torch.cat([prev_state.stoch, action], dim=-1)
        det = self.gru(gru_input, prev_state.det)
        prior_mean, prior_std = self._split_stats(self.prior_head(det))
        return {"det": det, "prior_mean": prior_mean, "prior_std": prior_std}

    def posterior(self, det: Tensor, obs: Tensor) -> Dict[str, Tensor]:
        obs_emb = self.obs_encoder(obs)
        stats = self.posterior_head(torch.cat([det, obs_emb], dim=-1))
        post_mean, post_std = self._split_stats(stats)
        return {"post_mean": post_mean, "post_std": post_std}

    def decode(self, state: RSSMState) -> Tuple[Tensor, Tensor, Tensor]:
        feat = torch.cat([state.det, state.stoch], dim=-1)
        obs = self.decoder(feat)
        reward = self.reward_head(feat).squeeze(-1)
        done_logit = self.done_head(feat).squeeze(-1)
        return obs, reward, done_logit

    def forward_step(self, obs: Tensor, action: Tensor, prev_state: RSSMState) -> Dict[str, Tensor | RSSMState]:
        trans = self.transition(prev_state, action)
        post = self.posterior(trans["det"], obs)

        prior_stoch = self._sample(trans["prior_mean"], trans["prior_std"])
        post_stoch = self._sample(post["post_mean"], post["post_std"])

        prior_state = RSSMState(det=trans["det"], stoch=prior_stoch)
        posterior_state = RSSMState(det=trans["det"], stoch=post_stoch)
        recon_obs, reward, done_logit = self.decode(posterior_state)

        return {
            "prior_state": prior_state,
            "posterior_state": posterior_state,
            "prior_mean": trans["prior_mean"],
            "prior_std": trans["prior_std"],
            "post_mean": post["post_mean"],
            "post_std": post["post_std"],
            "recon_obs": recon_obs,
            "reward": reward,
            "done_logit": done_logit,
        }

    def imagine_step(self, prev_state: RSSMState, action: Tensor, sample: bool = False) -> Dict[str, Tensor | RSSMState]:
        trans = self.transition(prev_state, action)
        stoch = self._sample(trans["prior_mean"], trans["prior_std"], sample=sample)
        next_state = RSSMState(det=trans["det"], stoch=stoch)
        pred_obs, reward, done_logit = self.decode(next_state)
        return {
            "state": next_state,
            "obs": pred_obs,
            "reward": reward,
            "done_logit": done_logit,
            "prior_mean": trans["prior_mean"],
            "prior_std": trans["prior_std"],
        }

    @staticmethod
    def kl_divergence(post_mean: Tensor, post_std: Tensor, prior_mean: Tensor, prior_std: Tensor) -> Tensor:
        """KL(q||p) between diagonal Gaussians."""
        var_ratio = (post_std.pow(2) + (post_mean - prior_mean).pow(2)) / (prior_std.pow(2) + 1e-8)
        kl = 0.5 * (2 * (torch.log(prior_std + 1e-8) - torch.log(post_std + 1e-8)) + var_ratio - 1)
        return kl.sum(dim=-1)

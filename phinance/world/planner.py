"""Planning utilities for world-model control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from phinance.world.model import RSSMState, WorldModelRSSM


@dataclass
class CEMConfig:
    horizon: int = 8
    candidates: int = 128
    elite_frac: float = 0.1
    iterations: int = 4
    action_low: float = 0.0
    action_high: float = 1.0


class CEMPlanner:
    """Cross-Entropy Method planner over imagined trajectories."""

    def __init__(self, model: WorldModelRSSM, config: CEMConfig | None = None) -> None:
        self.model = model
        self.config = config or CEMConfig()

    def plan(self, initial_state: RSSMState) -> np.ndarray:
        action_dim = self.model.config.action_dim
        mean = torch.full((self.config.horizon, action_dim), 0.5)
        std = torch.full((self.config.horizon, action_dim), 0.3)

        elite_k = max(1, int(self.config.candidates * self.config.elite_frac))
        device = initial_state.det.device

        for _ in range(self.config.iterations):
            noise = torch.randn(self.config.candidates, self.config.horizon, action_dim, device=device)
            actions = mean.to(device).unsqueeze(0) + noise * std.to(device).unsqueeze(0)
            actions = torch.clamp(actions, self.config.action_low, self.config.action_high)

            returns = self._evaluate(initial_state, actions)
            elite_idx = torch.topk(returns, k=elite_k).indices
            elite = actions[elite_idx]
            mean = elite.mean(dim=0).cpu()
            std = elite.std(dim=0).clamp_min(1e-3).cpu()

        return mean[0].numpy().astype(np.float32)

    def _evaluate(self, state: RSSMState, action_sequences: torch.Tensor) -> torch.Tensor:
        scores = []
        with torch.no_grad():
            for seq in action_sequences:
                s = RSSMState(det=state.det.clone(), stoch=state.stoch.clone())
                total = torch.tensor(0.0, device=seq.device)
                for action in seq:
                    out = self.model.imagine_step(s, action.unsqueeze(0), sample=False)
                    s = out["state"]
                    total = total + out["reward"].squeeze(0)
                scores.append(total)
        return torch.stack(scores)

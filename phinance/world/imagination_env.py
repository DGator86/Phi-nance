"""Environment wrapper that rolls out trajectories through a world model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from phinance.world.model import RSSMState, WorldModelRSSM

try:  # pragma: no cover - optional dependency
    from gymnasium.spaces import Box
except Exception:  # pragma: no cover
    class Box:  # type: ignore[override]
        def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype


@dataclass
class ImaginationEnvConfig:
    horizon: int = 64
    done_threshold: float = 0.7


class ImaginationEnv:
    """Gym-like environment driven by a trained world model."""

    def __init__(
        self,
        model: WorldModelRSSM,
        initial_observation: np.ndarray,
        config: ImaginationEnvConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device or torch.device("cpu")
        self.config = config or ImaginationEnvConfig()
        self.initial_observation = np.asarray(initial_observation, dtype=np.float32)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.model.config.obs_dim,), dtype=np.float32)
        self.action_space = Box(low=0.0, high=1.0, shape=(self.model.config.action_dim,), dtype=np.float32)

        self._state: RSSMState | None = None
        self._obs = self.initial_observation.copy()
        self._steps = 0

    def reset(self) -> np.ndarray:
        self._steps = 0
        self._obs = self.initial_observation.copy()
        self._state = self.model.initial_state(batch_size=1, device=self.device)
        zero_action = torch.zeros((1, self.model.config.action_dim), dtype=torch.float32, device=self.device)
        obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model.forward_step(obs_t, zero_action, self._state)
        self._state = out["posterior_state"]
        return self._obs.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset before step.")

        self._steps += 1
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model.imagine_step(self._state, action_t, sample=False)

        self._state = out["state"]
        obs = out["obs"].squeeze(0).cpu().numpy().astype(np.float32)
        reward = float(out["reward"].item())
        done_prob = float(torch.sigmoid(out["done_logit"]).item())
        done = done_prob > self.config.done_threshold or self._steps >= self.config.horizon
        self._obs = obs

        info = {"done_prob": done_prob, "steps": self._steps}
        return obs, reward, done, info

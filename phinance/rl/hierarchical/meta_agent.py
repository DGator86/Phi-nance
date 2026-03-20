"""Meta-agent policy network for option selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from phinance.rl.policy_networks import CategoricalPolicy


@dataclass
class MetaAgentConfig:
    hidden_size: int = 256
    architecture: str = "mlp"
    sequence_length: int = 16


class MetaAgent:
    """High-level controller selecting among low-level options."""

    def __init__(self, obs_dim: int, n_options: int, config: MetaAgentConfig | None = None) -> None:
        self.config = config or MetaAgentConfig()
        self.obs_dim = int(obs_dim)
        self.n_options = int(n_options)
        self.policy = CategoricalPolicy(
            obs_dim=self.obs_dim,
            n_actions=self.n_options,
            hidden_size=int(self.config.hidden_size),
            architecture=str(self.config.architecture),
            sequence_length=int(self.config.sequence_length),
        )

    def act(self, state: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.policy(tensor).squeeze(0)
            if deterministic:
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())

    def save(self, checkpoint: Path, extra: Dict[str, Any] | None = None) -> Path:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.policy.state_dict(),
            "obs_dim": self.obs_dim,
            "n_options": self.n_options,
            "architecture": self.config.architecture,
            "sequence_length": self.config.sequence_length,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, checkpoint)
        return checkpoint

    @classmethod
    def load(cls, checkpoint: str | Path) -> "MetaAgent":
        payload = torch.load(Path(checkpoint), map_location="cpu")
        cfg = MetaAgentConfig(
            architecture=str(payload.get("architecture", "mlp")),
            sequence_length=int(payload.get("sequence_length", 16)),
        )
        agent = cls(obs_dim=int(payload["obs_dim"]), n_options=int(payload["n_options"]), config=cfg)
        agent.policy.load_state_dict(payload["model_state_dict"])
        agent.policy.eval()
        return agent

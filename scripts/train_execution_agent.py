"""Train the RL execution policy using AReaL PPO when available."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from phinance.rl.execution_env import ExecutionEnv, ExecutionEnvConfig

logger = logging.getLogger(__name__)


class GaussianPolicy(nn.Module):
    """Simple Gaussian MLP policy for two continuous actions."""

    def __init__(self, obs_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_size, 2)
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = torch.sigmoid(self.mean_head(x))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = self.forward(tensor)
            if deterministic:
                action = mean
            else:
                action = torch.normal(mean, std)
        return torch.clamp(action.squeeze(0), 0.0, 1.0).cpu().numpy()


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> ExecutionEnv:
    env_cfg = ExecutionEnvConfig(**config.get("environment", {}))
    return ExecutionEnv(config=env_cfg)


def train_with_fallback_loop(config: Dict[str, Any], output_dir: Path) -> Path:
    """Fallback mini trainer used when AReaL is not available."""
    env = _build_env(config)
    policy = GaussianPolicy(obs_dim=env.observation_space.shape[0], hidden_size=config["model"]["hidden_size"])
    episodes = int(config["training"]["episodes_smoke"])

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy.act(state, deterministic=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        logger.info("Fallback episode %d reward=%0.5f", episode + 1, total_reward)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save({"model_state_dict": policy.state_dict(), "obs_dim": env.observation_space.shape[0]}, checkpoint)
    return checkpoint


def train_with_areal(config: Dict[str, Any], output_dir: Path) -> Path:
    """Train using AReaL AsyncTrainer + PPO."""
    try:
        from areal.rl import AsyncTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("AReaL is not installed. Install it or run fallback smoke training.") from exc

    env = _build_env(config)
    policy = GaussianPolicy(obs_dim=env.observation_space.shape[0], hidden_size=config["model"]["hidden_size"])
    trainer = AsyncTrainer(
        env=env,
        policy=policy,
        algorithm="ppo",
        total_timesteps=int(config["training"]["total_timesteps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        gamma=float(config["training"]["gamma"]),
        tensorboard_log=str(output_dir / "tb"),
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save({"model_state_dict": policy.state_dict(), "obs_dim": env.observation_space.shape[0]}, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train execution RL agent")
    parser.add_argument("--config", type=Path, default=Path("configs/execution_agent.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/execution_agent"))
    parser.add_argument("--fallback", action="store_true", help="Use fallback smoke loop instead of AReaL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = _load_config(args.config)
    if args.fallback:
        checkpoint = train_with_fallback_loop(config, args.output)
    else:
        checkpoint = train_with_areal(config, args.output)

    logger.info("Saved execution policy checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

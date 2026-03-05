"""Train the Risk Monitor RL policy using AReaL when available."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from phinance.rl.risk_monitor_env import RiskMonitorEnv, RiskMonitorEnvConfig, RISK_PROFILES


class RiskMonitorPolicy(nn.Module):
    """Categorical MLP policy for discrete risk-profile actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> RiskMonitorEnv:
    env_cfg = RiskMonitorEnvConfig(**config.get("environment", {}))
    return RiskMonitorEnv(config=env_cfg)


def train_with_fallback(config: Dict[str, Any], output_dir: Path) -> Path:
    env = _build_env(config)
    policy = RiskMonitorPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=config["model"].get("hidden_size", 256),
    )

    episodes = int(config["training"].get("episodes_smoke", 5))
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = int(np.random.randint(0, env.action_space.n))
            state, _, done, _ = env.step(action)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": env.observation_space.shape[0],
            "n_actions": env.action_space.n,
            "profiles": RISK_PROFILES,
        },
        checkpoint,
    )
    return checkpoint


def train_with_areal(config: Dict[str, Any], output_dir: Path) -> Path:
    from areal.rl import AsyncTrainer  # type: ignore

    env = _build_env(config)
    policy = RiskMonitorPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=config["model"].get("hidden_size", 256),
    )

    trainer = AsyncTrainer(
        env=env,
        policy=policy,
        algorithm=config["training"].get("algorithm", "ppo"),
        total_timesteps=int(config["training"]["total_timesteps"]),
        learning_rate=float(config["training"].get("learning_rate", 0.0003)),
        gamma=float(config["training"].get("gamma", 0.99)),
        tensorboard_log=str(output_dir / "tb"),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": env.observation_space.shape[0],
            "n_actions": env.action_space.n,
            "profiles": RISK_PROFILES,
        },
        checkpoint,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train risk monitor RL agent")
    parser.add_argument("--config", type=Path, default=Path("configs/risk_monitor_agent.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/risk_monitor_agent"))
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.fallback:
        train_with_fallback(config, args.output)
    else:
        train_with_areal(config, args.output)


if __name__ == "__main__":
    main()

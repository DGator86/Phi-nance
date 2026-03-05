"""Train the Strategy R&D RL policy using AReaL when available."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from phinance.rl.strategy_rd_env import StrategyRDEnv, StrategyRDEnvConfig

logger = logging.getLogger(__name__)


class StrategyRDPolicy(nn.Module):
    """Categorical MLP policy for discrete strategy-template actions."""

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

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(tensor).squeeze(0)
            if deterministic:
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            return int(sample.item())


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> StrategyRDEnv:
    env_cfg = StrategyRDEnvConfig(**config.get("environment", {}))
    return StrategyRDEnv(config=env_cfg)


def train_with_fallback_loop(config: Dict[str, Any], output_dir: Path) -> Path:
    """Fallback smoke trainer used when AReaL is unavailable."""
    env = _build_env(config)
    policy = StrategyRDPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=config["model"]["hidden_size"],
    )

    episodes = int(config["training"]["episodes_smoke"])
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = int(np.random.randint(0, env.action_space.n))
            state, reward, done, _ = env.step(action)
            total_reward += reward
        logger.info("Fallback episode %d reward=%0.5f", episode + 1, total_reward)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": env.observation_space.shape[0],
            "n_actions": env.action_space.n,
            "templates": env.templates,
        },
        checkpoint,
    )
    return checkpoint


def train_with_areal(config: Dict[str, Any], output_dir: Path) -> Path:
    """Train using AReaL AsyncTrainer + PPO for discrete policy."""
    try:
        from areal.rl import AsyncTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("AReaL is not installed. Install it or run fallback smoke training.") from exc

    env = _build_env(config)
    policy = StrategyRDPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=config["model"]["hidden_size"],
    )
    trainer = AsyncTrainer(
        env=env,
        policy=policy,
        algorithm=config["training"].get("algorithm", "ppo"),
        total_timesteps=int(config["training"]["total_timesteps"]),
        learning_rate=float(config["training"]["learning_rate"]),
        gamma=float(config["training"]["gamma"]),
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
            "templates": env.templates,
        },
        checkpoint,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strategy R&D RL agent")
    parser.add_argument("--config", type=Path, default=Path("configs/strategy_rd_agent.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/strategy_rd_agent"))
    parser.add_argument("--fallback", action="store_true", help="Use fallback smoke loop instead of AReaL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = _load_config(args.config)
    if args.fallback:
        checkpoint = train_with_fallback_loop(config, args.output)
    else:
        checkpoint = train_with_areal(config, args.output)

    logger.info("Saved strategy R&D policy checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

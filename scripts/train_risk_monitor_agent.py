"""Train the Risk Monitor RL policy using AReaL when available."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from phinance.rl.policy_networks import CategoricalPolicy
from phinance.rl.risk_monitor_env import RISK_PROFILES, RiskMonitorEnv, RiskMonitorEnvConfig


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> RiskMonitorEnv:
    env_cfg = RiskMonitorEnvConfig(**config.get("environment", {}))
    return RiskMonitorEnv(config=env_cfg)


def train_with_fallback(config: Dict[str, Any], output_dir: Path) -> Path:
    env = _build_env(config)
    model_cfg = config.get("model", {})
    policy = CategoricalPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        architecture=str(model_cfg.get("architecture", "mlp")),
        sequence_length=int(model_cfg.get("sequence_length", 16)),
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
            "architecture": str(model_cfg.get("architecture", "mlp")),
            "sequence_length": int(model_cfg.get("sequence_length", 16)),
        },
        checkpoint,
    )
    return checkpoint


def train_with_areal(config: Dict[str, Any], output_dir: Path) -> Path:
    from areal.rl import AsyncTrainer  # type: ignore

    env = _build_env(config)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    policy = CategoricalPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        architecture=str(model_cfg.get("architecture", "mlp")),
        sequence_length=int(model_cfg.get("sequence_length", 16)),
    )

    trainer = AsyncTrainer(
        env=env,
        policy=policy,
        algorithm=str(training_cfg.get("algorithm", "ppo")),
        total_timesteps=int(training_cfg["total_timesteps"]),
        learning_rate=float(training_cfg.get("learning_rate", 0.0003)),
        gamma=float(training_cfg.get("gamma", 0.99)),
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
            "architecture": str(model_cfg.get("architecture", "mlp")),
            "sequence_length": int(model_cfg.get("sequence_length", 16)),
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

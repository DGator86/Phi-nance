"""Train the Strategy R&D RL policy using AReaL when available."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from phinance.rl.policy_networks import CategoricalPolicy
from phinance.rl.strategy_rd_env import StrategyRDEnv, StrategyRDEnvConfig

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> StrategyRDEnv:
    env_cfg = StrategyRDEnvConfig(**config.get("environment", {}))
    return StrategyRDEnv(config=env_cfg)


def _act(policy: CategoricalPolicy, state: np.ndarray, deterministic: bool = False) -> int:
    with torch.no_grad():
        tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = policy(tensor).squeeze(0)
        if deterministic:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


def train_with_fallback_loop(config: Dict[str, Any], output_dir: Path) -> Path:
    """Fallback mini trainer used when AReaL is not available."""
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

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = _act(policy, state, deterministic=False)
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
            "architecture": str(model_cfg.get("architecture", "mlp")),
            "sequence_length": int(model_cfg.get("sequence_length", 16)),
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
            "templates": env.templates,
            "architecture": str(model_cfg.get("architecture", "mlp")),
            "sequence_length": int(model_cfg.get("sequence_length", 16)),
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

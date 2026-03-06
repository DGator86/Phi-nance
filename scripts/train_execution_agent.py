"""Train the RL execution policy using AReaL PPO/SAC/TD3 when available."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from phinance.rl.execution_env import ExecutionEnv, ExecutionEnvConfig
from phinance.rl.policy_networks import GaussianPolicy
from phinance.world.integration import (
    build_execution_training_env,
    estimate_world_model_confidence,
    should_fallback_to_model_free,
)

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> ExecutionEnv:
    env_cfg = ExecutionEnvConfig(**config.get("environment", {}))
    return ExecutionEnv(config=env_cfg)


def train_with_fallback_loop(config: Dict[str, Any], output_dir: Path, world_model_path: Path | None = None) -> Path:
    """Fallback mini trainer used when AReaL is not available."""
    env, using_world_model = build_execution_training_env(config, world_model_path)
    if using_world_model:
        recent_losses = np.asarray(config.get("world_model", {}).get("recent_losses", [0.35]), dtype=np.float32)
        confidence = estimate_world_model_confidence(recent_losses)
        if should_fallback_to_model_free(confidence, threshold=float(config.get("world_model", {}).get("confidence_threshold", 0.55))):
            logger.warning("World model confidence %.3f below threshold; switching to model-free fallback.", confidence)
            env = _build_env(config)
            using_world_model = False
        else:
            logger.info("Using imagination environment (confidence=%.3f)", confidence)
    model_cfg = config.get("model", {})
    policy = GaussianPolicy(
        obs_dim=env.observation_space.shape[0],
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
            with torch.no_grad():
                tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                mean, std = policy(tensor)
                action = torch.normal(mean, std)
                action = torch.clamp(action.squeeze(0), 0.0, 1.0).cpu().numpy()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        logger.info("Fallback episode %d reward=%0.5f", episode + 1, total_reward)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "architecture": str(model_cfg.get("architecture", "mlp")),
            "sequence_length": int(model_cfg.get("sequence_length", 16)),
        },
        checkpoint,
    )
    return checkpoint


def train_with_areal(config: Dict[str, Any], output_dir: Path, world_model_path: Path | None = None) -> Path:
    """Train using AReaL AsyncTrainer and selected algorithm."""
    try:
        from areal.rl import AsyncTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("AReaL is not installed. Install it or run fallback smoke training.") from exc

    env, using_world_model = build_execution_training_env(config, world_model_path)
    if using_world_model:
        logger.info("Training with imagination environment backed by world model %s", world_model_path)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    policy = GaussianPolicy(
        obs_dim=env.observation_space.shape[0],
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
            "action_dim": env.action_space.shape[0],
            "architecture": str(model_cfg.get("architecture", "mlp")),
            "sequence_length": int(model_cfg.get("sequence_length", 16)),
        },
        checkpoint,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train execution RL agent")
    parser.add_argument("--config", type=Path, default=Path("configs/execution_agent.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/execution_agent"))
    parser.add_argument("--fallback", action="store_true", help="Use fallback smoke loop instead of AReaL")
    parser.add_argument("--world-model-path", type=Path, default=None, help="Optional checkpoint for imagination training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = _load_config(args.config)
    if args.fallback:
        checkpoint = train_with_fallback_loop(config, args.output, args.world_model_path)
    else:
        checkpoint = train_with_areal(config, args.output, args.world_model_path)

    logger.info("Saved execution policy checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

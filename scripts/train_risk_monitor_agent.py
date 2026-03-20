"""Train the Risk Monitor RL policy using AReaL when available."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from phinance.rl.policy_networks import CategoricalPolicy
from phinance.rl.risk_monitor_env import RISK_PROFILES, RiskMonitorEnv, RiskMonitorEnvConfig
from phinance.rl.training_utils import get_runtime_config, load_optimisation_config, move_policy_to_device

logger = logging.getLogger(__name__)

def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _build_env(config: Dict[str, Any]) -> RiskMonitorEnv:
    env_cfg = RiskMonitorEnvConfig(**config.get("environment", {}))
    return RiskMonitorEnv(config=env_cfg)


def train_with_fallback(
    config: Dict[str, Any],
    output_dir: Path,
    optim_cfg: Dict[str, Any] | None = None,
    tracker: Any = None,
) -> tuple[Path, dict[str, float]]:
    env = _build_env(config)
    model_cfg = config.get("model", {})
    runtime_cfg = get_runtime_config(optim_cfg or {"rl_optimisation": {}})
    policy = CategoricalPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        architecture=str(model_cfg.get("architecture", "mlp")),
        sequence_length=int(model_cfg.get("sequence_length", 16)),
    )
    policy, _ = move_policy_to_device(policy, runtime_cfg)

    episodes = int(config["training"].get("episodes_smoke", 5))
    final_reward = 0.0
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = int(np.random.randint(0, env.action_space.n))
            state, reward, done, _ = env.step(action)
            total_reward += reward
        final_reward = float(total_reward)
        logger.info("Fallback episode %d reward=%0.5f", episode + 1, total_reward)
        if tracker is not None:
            tracker.log_metrics({"episode_reward": float(total_reward)}, step=episode + 1)

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
    return checkpoint, {"final_episode_reward": final_reward, "episodes": float(episodes)}


def train_with_areal(
    config: Dict[str, Any],
    output_dir: Path,
    optim_cfg: Dict[str, Any] | None = None,
    tracker: Any = None,
) -> tuple[Path, dict[str, float]]:
    from areal.rl import AsyncTrainer  # type: ignore

    env = _build_env(config)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    runtime_cfg = get_runtime_config(optim_cfg or {"rl_optimisation": {}})
    policy = CategoricalPolicy(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        architecture=str(model_cfg.get("architecture", "mlp")),
        sequence_length=int(model_cfg.get("sequence_length", 16)),
    )
    policy, _ = move_policy_to_device(policy, runtime_cfg)

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
    if tracker is not None:
        tracker.log_metrics({"total_timesteps": float(training_cfg["total_timesteps"])})

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
    return checkpoint, {"total_timesteps": float(training_cfg["total_timesteps"])}


def train_risk_monitor_agent(
    config: str = "configs/risk_monitor_agent.yaml",
    optim_config: str = "configs/rl_optimisation_config.yaml",
    output: str = "models/risk_monitor_agent",
    fallback: bool = True,
    tracker: Any = None,
) -> dict[str, float]:
    config_path = Path(config)
    optim_config_path = Path(optim_config)
    output_path = Path(output)

    cfg = _load_config(config_path)
    optim_cfg = load_optimisation_config(optim_config_path)
    if tracker is not None:
        tracker.log_params(
            {
                "config": str(config_path),
                "optim_config": str(optim_config_path),
                "fallback": bool(fallback),
            }
        )

    if fallback:
        checkpoint, metrics = train_with_fallback(cfg, output_path, optim_cfg, tracker=tracker)
    else:
        checkpoint, metrics = train_with_areal(cfg, output_path, optim_cfg, tracker=tracker)

    if tracker is not None:
        tracker.log_artifact(str(checkpoint))

    metrics["checkpoint_size_bytes"] = float(checkpoint.stat().st_size if checkpoint.exists() else 0)
    metrics["used_fallback"] = float(bool(fallback))
    return metrics


def run_experiment_target(
    config: str = "configs/risk_monitor_agent.yaml",
    optim_config: str = "configs/rl_optimisation_config.yaml",
    output: str = "models/risk_monitor_agent",
    fallback: bool = True,
    tracker: Any = None,
) -> dict[str, float]:
    return train_risk_monitor_agent(config, optim_config, output, fallback, tracker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train risk monitor RL agent")
    parser.add_argument("--config", type=Path, default=Path("configs/risk_monitor_agent.yaml"))
    parser.add_argument("--optim-config", type=Path, default=Path("configs/rl_optimisation_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/risk_monitor_agent"))
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = _load_config(args.config)
    optim_cfg = load_optimisation_config(args.optim_config)
    if args.fallback:
        checkpoint, _ = train_with_fallback(config, args.output, optim_cfg)
    else:
        checkpoint, _ = train_with_areal(config, args.output, optim_cfg)

    logger.info("Saved risk monitor checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

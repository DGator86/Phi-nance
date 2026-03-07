"""Training helpers for hierarchical meta-agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from phinance.rl.execution_env import ExecutionEnv, ExecutionEnvConfig
from phinance.rl.hierarchical.meta_agent import MetaAgent, MetaAgentConfig
from phinance.rl.hierarchical.meta_env import MetaEnv, MetaEnvConfig
from phinance.rl.hierarchical.wrappers import build_default_options
from phinance.rl.training_utils import get_runtime_config, load_optimisation_config, move_policy_to_device

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def build_meta_env(config: Dict[str, Any], optim_cfg: Dict[str, Any] | None = None) -> MetaEnv:
    env_data = dict(config.get("environment", {}))
    if optim_cfg:
        env_data["enable_numba"] = bool(optim_cfg.get("rl_optimisation", {}).get("numba", {}).get("enabled", False))
        env_data["enable_state_cache"] = bool(optim_cfg.get("rl_optimisation", {}).get("caching", {}).get("enabled", False))
    env_cfg = ExecutionEnvConfig(**env_data)
    base_env = ExecutionEnv(config=env_cfg)
    options = build_default_options(config.get("options", {}))
    meta_cfg = MetaEnvConfig(**config.get("meta_env", {}))
    return MetaEnv(env=base_env, options=options, config=meta_cfg)


def train_with_fallback_loop(config: Dict[str, Any], output_dir: Path, optim_cfg: Dict[str, Any] | None = None) -> Path:
    env = build_meta_env(config, optim_cfg)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    meta_agent = MetaAgent(
        obs_dim=env.observation_space.shape[0],
        n_options=env.action_space.n,
        config=MetaAgentConfig(
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            architecture=str(model_cfg.get("architecture", "mlp")),
            sequence_length=int(model_cfg.get("sequence_length", 16)),
        ),
    )
    runtime_cfg = get_runtime_config(optim_cfg or {"rl_optimisation": {}})
    move_policy_to_device(meta_agent.policy, runtime_cfg)

    episodes = int(train_cfg.get("episodes_smoke", 5))
    for episode in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action = int(np.random.randint(0, env.action_space.n))
            state, reward, done, _ = env.step(action)
            total += reward
        logger.info("Meta fallback episode %d reward=%0.4f", episode + 1, total)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    meta_agent.save(
        checkpoint,
        extra={"option_names": [option.name for option in env.options], "trainer": "fallback"},
    )
    return checkpoint


def train_with_areal(config: Dict[str, Any], output_dir: Path, optim_cfg: Dict[str, Any] | None = None) -> Path:
    try:
        from areal.rl import AsyncTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("AReaL is not installed. Run with fallback for smoke training.") from exc

    env = build_meta_env(config, optim_cfg)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    runtime_cfg = get_runtime_config(optim_cfg or {"rl_optimisation": {}})
    meta_agent = MetaAgent(
        obs_dim=env.observation_space.shape[0],
        n_options=env.action_space.n,
        config=MetaAgentConfig(
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            architecture=str(model_cfg.get("architecture", "mlp")),
            sequence_length=int(model_cfg.get("sequence_length", 16)),
        ),
    )
    move_policy_to_device(meta_agent.policy, runtime_cfg)

    trainer = AsyncTrainer(
        env=env,
        policy=meta_agent.policy,
        algorithm=str(train_cfg.get("algorithm", "ppo")),
        total_timesteps=int(train_cfg.get("total_timesteps", 10000)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        tensorboard_log=str(output_dir / "tb"),
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "latest.pt"
    meta_agent.save(
        checkpoint,
        extra={"option_names": [option.name for option in env.options], "trainer": "areal"},
    )
    return checkpoint

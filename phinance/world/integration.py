"""Integration helpers between world models and existing RL agents."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from phinance.rl.execution_env import ExecutionEnv, ExecutionEnvConfig
from phinance.world.imagination_env import ImaginationEnv, ImaginationEnvConfig
from phinance.world.trainer import WorldModelTrainer


def build_execution_training_env(
    config: Dict,
    world_model_path: Path | None = None,
) -> Tuple[object, bool]:
    """Return real ExecutionEnv or ImaginationEnv depending on world-model availability."""
    real_env = ExecutionEnv(config=ExecutionEnvConfig(**config.get("environment", {})))

    if world_model_path is None or not world_model_path.exists():
        return real_env, False

    model = WorldModelTrainer.load(world_model_path)
    seed_obs = real_env.reset()
    imag_env = ImaginationEnv(
        model=model,
        initial_observation=seed_obs,
        config=ImaginationEnvConfig(horizon=int(config.get("world_model", {}).get("imagination_horizon", 64))),
    )
    return imag_env, True


def should_fallback_to_model_free(model_confidence: float, threshold: float = 0.55) -> bool:
    """Simple confidence gate for model-based vs model-free routing."""
    return model_confidence < threshold


def estimate_world_model_confidence(recent_losses: np.ndarray) -> float:
    """Map recent reconstruction losses to [0,1] confidence score."""
    loss = float(np.mean(recent_losses)) if recent_losses.size else 1.0
    return float(np.exp(-loss))

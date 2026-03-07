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
from phinance.rl.experience_buffer import OptimisedExperienceBuffer
from phinance.rl.optimised_env_runner import RayEnvRunner, RayEnvRunnerConfig, transitions_to_arrays
from phinance.rl.policy_networks import GaussianPolicy
from phinance.rl.training_utils import (
    autocast_context,
    build_grad_scaler,
    get_runtime_config,
    load_optimisation_config,
    move_policy_to_device,
)
from phinance.utils.performance import PerformanceTracker, track_time
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


def _build_env(config: Dict[str, Any], optim_cfg: Dict[str, Any] | None = None) -> ExecutionEnv:
    env_data = dict(config.get("environment", {}))
    if optim_cfg:
        env_data["enable_numba"] = bool(optim_cfg.get("rl_optimisation", {}).get("numba", {}).get("enabled", False))
        env_data["enable_state_cache"] = bool(optim_cfg.get("rl_optimisation", {}).get("caching", {}).get("enabled", False))
    env_cfg = ExecutionEnvConfig(**env_data)
    return ExecutionEnv(config=env_cfg)


def _run_parallel_collection(config: Dict[str, Any], optim_cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    parallel_cfg = optim_cfg.get("rl_optimisation", {}).get("parallel_rollouts", {})
    runner = RayEnvRunner(
        env_factory=lambda: _build_env(config, optim_cfg),
        config=RayEnvRunnerConfig(
            num_workers=int(parallel_cfg.get("num_workers", 4)),
            use_local_mode=bool(parallel_cfg.get("local_mode", False)),
        ),
    )
    try:
        transitions = runner.run_steps(int(parallel_cfg.get("steps_per_collect", 256)))
        return transitions_to_arrays(transitions)
    finally:
        runner.shutdown()


def train_with_fallback_loop(
    config: Dict[str, Any],
    output_dir: Path,
    world_model_path: Path | None = None,
    optim_cfg: Dict[str, Any] | None = None,
) -> Path:
    """Fallback mini trainer used when AReaL is not available."""
    optim_cfg = optim_cfg or {"rl_optimisation": {}}
    env, using_world_model = build_execution_training_env(config, world_model_path)
    if using_world_model:
        recent_losses = np.asarray(config.get("world_model", {}).get("recent_losses", [0.35]), dtype=np.float32)
        confidence = estimate_world_model_confidence(recent_losses)
        if should_fallback_to_model_free(confidence, threshold=float(config.get("world_model", {}).get("confidence_threshold", 0.55))):
            logger.warning("World model confidence %.3f below threshold; switching to model-free fallback.", confidence)
            env = _build_env(config, optim_cfg)
            using_world_model = False
        else:
            logger.info("Using imagination environment (confidence=%.3f)", confidence)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    tracker = PerformanceTracker()
    runtime_cfg = get_runtime_config(optim_cfg)

    policy = GaussianPolicy(
        obs_dim=env.observation_space.shape[0],
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        architecture=str(model_cfg.get("architecture", "mlp")),
        sequence_length=int(model_cfg.get("sequence_length", 16)),
    )
    policy, device = move_policy_to_device(policy, runtime_cfg)
    scaler = build_grad_scaler(runtime_cfg)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg.get("learning_rate", 3e-4)))
    episodes = int(training_cfg.get("episodes_smoke", 5))

    buffer_cfg = optim_cfg.get("rl_optimisation", {}).get("experience_buffer", {})
    buffer = None
    if bool(buffer_cfg.get("enabled", False)):
        buffer = OptimisedExperienceBuffer(
            capacity=int(buffer_cfg.get("size", 100_000)),
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            prefetch=bool(buffer_cfg.get("prefetch", True)),
        )

    parallel_enabled = bool(optim_cfg.get("rl_optimisation", {}).get("parallel_rollouts", {}).get("enabled", False))
    for episode in range(episodes):
        with track_time(tracker, "episode"):
            state = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                with track_time(tracker, "policy_forward"):
                    with torch.no_grad():
                        tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        with autocast_context(runtime_cfg.mixed_precision, device.type):
                            mean, std = policy(tensor)
                            action = torch.normal(mean, std)
                            action = torch.clamp(action.squeeze(0), 0.0, 1.0).detach().cpu().numpy()
                with track_time(tracker, "env_step"):
                    next_state, reward, done, _ = env.step(action)
                if buffer is not None:
                    buffer.add_batch(
                        states=np.asarray([state]),
                        actions=np.asarray([action]),
                        rewards=np.asarray([reward], dtype=np.float32),
                        next_states=np.asarray([next_state]),
                        dones=np.asarray([done], dtype=np.float32),
                    )
                    if buffer.size >= int(buffer_cfg.get("batch_size", 128)):
                        with track_time(tracker, "buffer_sample"):
                            batch = buffer.sample(int(buffer_cfg.get("batch_size", 128)))
                        with track_time(tracker, "optim_step"):
                            optimizer.zero_grad(set_to_none=True)
                            sample_states = torch.tensor(batch.states, device=device)
                            with autocast_context(runtime_cfg.mixed_precision, device.type):
                                pred_mean, _ = policy(sample_states)
                                loss = (pred_mean ** 2).mean()
                            scaled = scaler.scale(loss)
                            scaled.backward()
                            scaler.step(optimizer)
                            scaler.update()
                state = next_state
                total_reward += reward
            logger.info("Fallback episode %d reward=%0.5f", episode + 1, total_reward)

    if parallel_enabled:
        with track_time(tracker, "ray_collect"):
            collected = _run_parallel_collection(config, optim_cfg)
            logger.info("Ray collected %d transitions", int(collected["states"].shape[0]))

    logger.info("Fallback optimisation profile:\n%s", tracker.as_markdown())
    if buffer is not None:
        buffer.close()

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


def train_with_areal(
    config: Dict[str, Any],
    output_dir: Path,
    world_model_path: Path | None = None,
    optim_cfg: Dict[str, Any] | None = None,
) -> Path:
    """Train using AReaL AsyncTrainer and selected algorithm."""
    try:
        from areal.rl import AsyncTrainer  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("AReaL is not installed. Install it or run fallback smoke training.") from exc

    optim_cfg = optim_cfg or {"rl_optimisation": {}}
    env, using_world_model = build_execution_training_env(config, world_model_path)
    if using_world_model:
        logger.info("Training with imagination environment backed by world model %s", world_model_path)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    runtime_cfg = get_runtime_config(optim_cfg)

    policy = GaussianPolicy(
        obs_dim=env.observation_space.shape[0],
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
    parser.add_argument("--optim-config", type=Path, default=Path("configs/rl_optimisation_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/execution_agent"))
    parser.add_argument("--fallback", action="store_true", help="Use fallback smoke loop instead of AReaL")
    parser.add_argument("--world-model-path", type=Path, default=None, help="Optional checkpoint for imagination training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = _load_config(args.config)
    optim_cfg = load_optimisation_config(args.optim_config)
    if args.fallback:
        checkpoint = train_with_fallback_loop(config, args.output, args.world_model_path, optim_cfg)
    else:
        checkpoint = train_with_areal(config, args.output, args.world_model_path, optim_cfg)

    logger.info("Saved execution policy checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

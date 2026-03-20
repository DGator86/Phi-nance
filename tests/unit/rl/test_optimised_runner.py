from __future__ import annotations

import numpy as np
import pytest

from phinance.rl.execution_env import ExecutionEnv, ExecutionEnvConfig
from phinance.rl.optimised_env_runner import RayEnvRunner, RayEnvRunnerConfig

ray = pytest.importorskip("ray")


def _factory() -> ExecutionEnv:
    return ExecutionEnv(config=ExecutionEnvConfig(episode_length=10, seed=11))


def test_ray_env_runner_collects_transitions():
    if ray.is_initialized():
        ray.shutdown()
    runner = RayEnvRunner(_factory, config=RayEnvRunnerConfig(num_workers=2, use_local_mode=True))
    transitions = runner.run_steps(20)
    runner.shutdown()

    assert len(transitions) == 20
    state, action, reward, next_state, done = transitions[0]
    assert isinstance(reward, float)
    assert state.shape == next_state.shape
    assert np.asarray(action).ndim == 1
    assert isinstance(done, bool)

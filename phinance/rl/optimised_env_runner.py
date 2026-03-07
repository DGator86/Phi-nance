"""Ray-backed environment rollout runner for opt-in parallel collection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Experience = Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]

try:  # pragma: no cover - optional dependency
    import ray
except Exception:  # pragma: no cover
    ray = None  # type: ignore[assignment]


@dataclass
class RayEnvRunnerConfig:
    num_workers: int = 4
    use_local_mode: bool = False


if ray is not None:  # pragma: no cover - import guard

    @ray.remote
    class _EnvActor:
        def __init__(self, env_factory: Callable[[], Any], seed: int) -> None:
            self.env = env_factory()
            self.rng = np.random.default_rng(seed)
            self.state = self.env.reset()

        def run_steps(self, num_steps: int) -> List[Experience]:
            transitions: List[Experience] = []
            for _ in range(num_steps):
                if hasattr(self.env.action_space, "sample"):
                    action = self.env.action_space.sample()
                else:
                    # Compatibility fallback for lightweight test stubs.
                    if hasattr(self.env.action_space, "n"):
                        action = int(self.rng.integers(0, int(self.env.action_space.n)))
                    else:
                        action = self.rng.random(2, dtype=np.float32)
                next_state, reward, done, _ = self.env.step(action)
                transitions.append(
                    (
                        np.asarray(self.state, dtype=np.float32),
                        np.asarray(action, dtype=np.float32),
                        float(reward),
                        np.asarray(next_state, dtype=np.float32),
                        bool(done),
                    )
                )
                self.state = self.env.reset() if done else next_state
            return transitions


class RayEnvRunner:
    """Collect transition tuples from multiple environment workers."""

    def __init__(self, env_factory: Callable[[], Any], config: Optional[RayEnvRunnerConfig] = None, seed: int = 7) -> None:
        if ray is None:
            raise RuntimeError("Ray is not available. Install ray[default] to use RayEnvRunner.")
        self.config = config or RayEnvRunnerConfig()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, local_mode=self.config.use_local_mode)
        self._workers = [_EnvActor.remote(env_factory, seed + i) for i in range(self.config.num_workers)]

    def run_steps(self, num_steps: int) -> List[Experience]:
        per_worker = max(1, math.ceil(num_steps / max(len(self._workers), 1)))
        futures = [worker.run_steps.remote(per_worker) for worker in self._workers]
        batches: Sequence[List[Experience]] = ray.get(futures)

        merged: List[Experience] = []
        for batch in batches:
            merged.extend(batch)
        return merged[:num_steps]

    def shutdown(self) -> None:
        if ray is not None and ray.is_initialized():
            ray.shutdown()


def transitions_to_arrays(transitions: Sequence[Experience]) -> Dict[str, np.ndarray]:
    states = np.asarray([t[0] for t in transitions], dtype=np.float32)
    actions = np.asarray([t[1] for t in transitions], dtype=np.float32)
    rewards = np.asarray([t[2] for t in transitions], dtype=np.float32)
    next_states = np.asarray([t[3] for t in transitions], dtype=np.float32)
    dones = np.asarray([t[4] for t in transitions], dtype=np.float32)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
    }

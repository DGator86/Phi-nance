"""Optimised replay buffer with pre-allocation and optional prefetch sampling."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ExperienceBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class OptimisedExperienceBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        prefetch: bool = False,
        seed: int = 7,
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.prefetch = bool(prefetch)
        self._rng = np.random.default_rng(seed)

        self._states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._size = 0
        self._ptr = 0
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        self._prefetch_future: Optional[Future[ExperienceBatch]] = None

    @property
    def size(self) -> int:
        return self._size

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        batch_size = int(states.shape[0])
        if batch_size == 0:
            return

        with self._lock:
            idx = (np.arange(batch_size) + self._ptr) % self.capacity
            self._states[idx] = states
            self._actions[idx] = actions
            self._rewards[idx] = rewards
            self._next_states[idx] = next_states
            self._dones[idx] = dones
            self._ptr = (self._ptr + batch_size) % self.capacity
            self._size = min(self.capacity, self._size + batch_size)

    def _sample_now(self, batch_size: int) -> ExperienceBatch:
        with self._lock:
            if self._size == 0:
                raise ValueError("Cannot sample from an empty experience buffer")
            idx = self._rng.integers(0, self._size, size=int(batch_size))
            return ExperienceBatch(
                states=self._states[idx].copy(),
                actions=self._actions[idx].copy(),
                rewards=self._rewards[idx].copy(),
                next_states=self._next_states[idx].copy(),
                dones=self._dones[idx].copy(),
            )

    def sample(self, batch_size: int) -> ExperienceBatch:
        if not self.prefetch or self._executor is None:
            return self._sample_now(batch_size)

        if self._prefetch_future is None:
            current = self._sample_now(batch_size)
            self._prefetch_future = self._executor.submit(self._sample_now, batch_size)
            return current

        current = self._prefetch_future.result()
        self._prefetch_future = self._executor.submit(self._sample_now, batch_size)
        return current

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)

    def as_dict(self) -> Dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "prefetch": int(self.prefetch),
        }

from __future__ import annotations

import numpy as np

from phinance.rl.experience_buffer import OptimisedExperienceBuffer


def test_experience_buffer_add_and_sample():
    buffer = OptimisedExperienceBuffer(capacity=16, obs_dim=4, action_dim=2, prefetch=False)
    states = np.random.randn(8, 4).astype(np.float32)
    actions = np.random.randn(8, 2).astype(np.float32)
    rewards = np.random.randn(8).astype(np.float32)
    next_states = np.random.randn(8, 4).astype(np.float32)
    dones = np.zeros(8, dtype=np.float32)

    buffer.add_batch(states, actions, rewards, next_states, dones)
    batch = buffer.sample(4)

    assert buffer.size == 8
    assert batch.states.shape == (4, 4)
    assert batch.actions.shape == (4, 2)


def test_experience_buffer_prefetch_roundtrip():
    buffer = OptimisedExperienceBuffer(capacity=32, obs_dim=3, action_dim=1, prefetch=True)
    for _ in range(3):
        buffer.add_batch(
            np.random.randn(10, 3).astype(np.float32),
            np.random.randn(10, 1).astype(np.float32),
            np.random.randn(10).astype(np.float32),
            np.random.randn(10, 3).astype(np.float32),
            np.zeros(10, dtype=np.float32),
        )
    first = buffer.sample(6)
    second = buffer.sample(6)
    buffer.close()

    assert first.states.shape == second.states.shape == (6, 3)

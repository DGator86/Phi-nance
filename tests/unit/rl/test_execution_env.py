from __future__ import annotations

import numpy as np
import pandas as pd

from phinance.agents.execution import ExecutionAgent
from phinance.rl.execution_env import ExecutionEnv, ExecutionEnvConfig


def _sample_data(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="min")
    base = 100 + np.linspace(0, 1, rows)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base + np.sin(np.linspace(0, 6, rows)) * 0.02,
            "volume": np.linspace(10000, 20000, rows),
        },
        index=idx,
    )


def test_execution_env_runs_multiple_steps() -> None:
    env = ExecutionEnv(data=_sample_data(), config=ExecutionEnvConfig(episode_length=10))
    obs = env.reset()
    assert obs.shape == (9,)

    done = False
    steps = 0
    while not done and steps < 20:
        obs, reward, done, info = env.step(np.array([0.4, 0.5]))
        assert isinstance(reward, float)
        assert "executed_shares" in info
        steps += 1

    assert done


def test_state_is_normalized_between_zero_and_one() -> None:
    env = ExecutionEnv(data=_sample_data(), config=ExecutionEnvConfig(episode_length=12))
    obs = env.reset()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_execution_agent_action_bounds_with_twap_fallback() -> None:
    agent = ExecutionAgent(use_rl=False)
    order = {"qty": 1000.0, "remaining_shares": 600.0, "remaining_steps": 6, "horizon_steps": 10}
    decision = agent.execute_order(order, _sample_data())
    assert 0.0 <= decision.urgency <= 1.0
    assert decision.shares_to_trade == 100.0

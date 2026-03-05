from __future__ import annotations

import numpy as np
import pandas as pd

from phinance.agents.strategy_rd import StrategyRDAgent
from phinance.rl.strategy_rd_env import StrategyRDEnv, StrategyRDEnvConfig


def _sample_data(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="D")
    base = 100 + np.linspace(0, 4, rows)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + np.sin(np.linspace(0, 12, rows)) * 0.5,
            "volume": np.linspace(1_000_000, 2_500_000, rows),
        },
        index=idx,
    )


def test_strategy_rd_env_init_and_reset() -> None:
    env = StrategyRDEnv(data=_sample_data(), config=StrategyRDEnvConfig(episode_length=4))
    obs = env.reset()
    assert obs.shape == (7,)
    assert env.action_space.n > 0


def test_strategy_rd_env_step_and_reward_match_lookup() -> None:
    env = StrategyRDEnv(data=_sample_data(), config=StrategyRDEnvConfig(episode_length=3))
    env.reset()

    obs, reward, done, info = env.step(0)
    assert isinstance(reward, float)
    assert np.isclose(reward, env.template_rewards[0])
    assert "template" in info
    assert obs.shape == (7,)
    assert not done


def test_strategy_rd_state_normalization_and_agent_fallback() -> None:
    env = StrategyRDEnv(data=_sample_data(), config=StrategyRDEnvConfig(episode_length=2))
    obs = env.reset()
    assert np.all(obs >= -1.0)
    assert np.all(obs <= 1.0)

    agent = StrategyRDAgent(use_rl=False)
    proposal = agent.propose_strategy(
        {
            "regime": "bull",
            "volatility": 0.2,
            "recent_performance": 0.1,
            "exploration_count": 0.3,
            "best_sharpe": 0.4,
        }
    )
    assert "name" in proposal
    assert "params" in proposal

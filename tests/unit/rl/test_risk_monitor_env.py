from __future__ import annotations

import numpy as np
import pandas as pd

from phinance.agents.risk_monitor import RiskMonitorAgent
from phinance.rl.risk_monitor_env import RiskMonitorEnv, RiskMonitorEnvConfig


def _sample_data(rows: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    base = 400 + np.linspace(0, 20, rows)
    close = base + np.sin(np.linspace(0, 30, rows)) * 2.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1_200_000,
        },
        index=idx,
    )


def test_risk_monitor_env_reset_and_step() -> None:
    env = RiskMonitorEnv(data=_sample_data(), config=RiskMonitorEnvConfig(episode_length=6, step_days=3))
    obs = env.reset()
    assert obs.shape == (14,)

    obs, reward, done, info = env.step(2)
    assert isinstance(reward, float)
    assert "profile" in info
    assert obs.shape == (14,) or done


def test_risk_monitor_env_catastrophic_termination() -> None:
    crashing = _sample_data().copy()
    crashing.loc[:, "close"] = np.linspace(100, 1, len(crashing))
    env = RiskMonitorEnv(
        data=crashing,
        config=RiskMonitorEnvConfig(episode_length=20, step_days=5, catastrophic_drawdown=0.05),
    )
    env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(3)
    assert info["catastrophic"] is True


def test_risk_monitor_agent_fallback_profile() -> None:
    agent = RiskMonitorAgent(use_rl=False)
    profile = agent.get_risk_limits(
        {"drawdown": 0.1, "var_95": 0.02, "beta": 1.1, "leverage_ratio": 0.25, "rebalance_age": 0.1},
        {"regime": "bull", "volatility": 0.2, "exploration_count": 0.3},
    )
    assert profile["name"] == "moderate"

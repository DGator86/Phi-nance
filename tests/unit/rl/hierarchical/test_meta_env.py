from __future__ import annotations

import numpy as np

from phinance.rl.hierarchical.meta_env import MetaEnv, MetaEnvConfig
from phinance.rl.hierarchical.options import Option


class _ToyEnv:
    def __init__(self) -> None:
        self.t = 0
        self.market_state = {"regime_value": 0.3, "volatility": 0.2}
        self.portfolio_state = {"sharpe": 0.4, "drawdown": 0.1}

    def reset(self):
        self.t = 0
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        value = float(np.asarray(action).reshape(-1)[0])
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), value, done, {"toy": True}


class _Policy:
    def __init__(self, action: int):
        self.action = action

    def act(self, state, deterministic: bool = True):  # noqa: ANN001,ARG002
        return self.action


def _option(name: str, action: int) -> Option:
    return Option(
        name=name,
        policy=_Policy(action),
        initiation_condition=lambda ctx: True,
        termination_condition=lambda ctx, info: False,
        max_steps=2,
    )


def test_meta_env_exposes_discrete_option_actions() -> None:
    env = MetaEnv(_ToyEnv(), [_option("a", 1), _option("b", 2)], MetaEnvConfig(option_horizon=1))
    obs = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    assert env.action_space.n == 2


def test_meta_env_runs_option_and_shapes_reward() -> None:
    env = MetaEnv(_ToyEnv(), [_option("a", 1), _option("idle", 0)], MetaEnvConfig(option_horizon=1, switch_penalty=0.5))
    _ = env.reset()
    _, reward, _, info = env.step(0)
    assert isinstance(reward, float)
    assert "active_option" in info

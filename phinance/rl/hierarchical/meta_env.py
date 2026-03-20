"""Meta-environment exposing option selection actions for hierarchical RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from phinance.rl.hierarchical.options import Option

try:  # pragma: no cover
    from gymnasium.spaces import Box, Discrete
except Exception:  # pragma: no cover
    class Box:  # type: ignore[override]
        def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class Discrete:  # type: ignore[override]
        def __init__(self, n: int) -> None:
            self.n = int(n)


@dataclass
class MetaEnvConfig:
    option_horizon: int = 3
    switch_penalty: float = 0.01
    inactivity_penalty: float = 0.001


class MetaEnv:
    """Wrapper that turns low-level env interaction into option-level control."""

    def __init__(self, env: Any, options: list[Option], config: MetaEnvConfig | None = None) -> None:
        if not options:
            raise ValueError("MetaEnv requires at least one option")
        self.env = env
        self.options = options
        self.config = config or MetaEnvConfig()

        self.action_space = Discrete(len(options))
        obs_dim = 6 + len(options)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.current_option_idx = len(options) - 1
        self.option_elapsed = 0
        self.switch_count = 0
        self.last_reward = 0.0
        self.global_step = 0
        self.last_env_obs: Any = None

    def _coerce_env_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32).reshape(-1)
        elif isinstance(obs, (list, tuple)):
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        else:
            arr = np.zeros(4, dtype=np.float32)
        if arr.size < 4:
            arr = np.pad(arr, (0, 4 - arr.size))
        return arr[:4]

    def _context(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "option_elapsed": self.option_elapsed,
            "market_state": self._extract_market_state(),
            "portfolio_state": self._extract_portfolio_state(),
            "order": getattr(self.env, "current_order", {}),
            "strategy_interval": getattr(self.env, "strategy_interval", 10),
        }

    def _extract_market_state(self) -> Dict[str, float]:
        return dict(getattr(self.env, "market_state", {}))

    def _extract_portfolio_state(self) -> Dict[str, float]:
        return dict(getattr(self.env, "portfolio_state", {}))

    def _build_meta_state(self, env_obs: Any) -> np.ndarray:
        core = self._coerce_env_obs(env_obs)
        market_state = self._extract_market_state()
        portfolio_state = self._extract_portfolio_state()
        regime_value = float(market_state.get("regime_value", 0.0))
        volatility = float(np.clip(market_state.get("volatility", 0.0), 0.0, 1.0))
        sharpe = float(np.clip(portfolio_state.get("sharpe", 0.0), -2.0, 2.0) / 2.0)
        drawdown = float(np.clip(portfolio_state.get("drawdown", 0.0), 0.0, 1.0))

        active = np.zeros(len(self.options), dtype=np.float32)
        active[self.current_option_idx] = 1.0
        switch_age = np.array([np.clip(self.option_elapsed / max(self.config.option_horizon, 1), 0.0, 1.0)], dtype=np.float32)
        aux = np.array([regime_value, volatility, sharpe, drawdown], dtype=np.float32)
        return np.concatenate([core[:1], aux, switch_age, active]).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.current_option_idx = len(self.options) - 1
        self.option_elapsed = 0
        self.switch_count = 0
        self.last_reward = 0.0
        self.global_step = 0
        obs = self.env.reset()
        self.last_env_obs = obs
        return self._build_meta_state(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, Dict[str, Any]]:
        desired = int(action)
        if desired < 0 or desired >= len(self.options):
            raise ValueError(f"Action out of bounds for options: {desired}")

        context = self._context()
        active_option = self.options[self.current_option_idx]
        should_switch = desired != self.current_option_idx and self.options[desired].can_initiate(context)
        can_interrupt = active_option.can_interrupt or active_option.should_terminate(context, {})
        if should_switch and can_interrupt:
            self.current_option_idx = desired
            self.option_elapsed = 0
            self.switch_count += 1

        option = self.options[self.current_option_idx]
        horizon = max(1, int(self.config.option_horizon))
        cumulative_reward = 0.0
        done = False
        info: Dict[str, Any] = {"active_option": option.name, "switch_count": self.switch_count}

        for _ in range(horizon):
            context = self._context()
            option_action = option.act(context, deterministic=False)
            env_action = option_action
            if isinstance(option_action, (int, float)):
                env_action = np.array([float(option_action), 0.5], dtype=np.float32)
            elif isinstance(option_action, (list, tuple, np.ndarray)):
                arr = np.asarray(option_action, dtype=np.float32).reshape(-1)
                if arr.size == 0:
                    env_action = np.array([0.0, 0.5], dtype=np.float32)
                elif arr.size == 1:
                    env_action = np.array([float(arr[0]), 0.5], dtype=np.float32)
                else:
                    env_action = arr[:2]
            else:
                env_action = np.array([0.0, 0.0], dtype=np.float32)
            env_obs, reward, done, env_info = self.env.step(env_action)
            info.update(env_info)
            cumulative_reward += float(reward)
            self.option_elapsed += 1
            self.global_step += 1
            self.last_env_obs = env_obs
            if done or option.should_terminate(self._context(), env_info):
                break

        shaped_reward = cumulative_reward
        if should_switch:
            shaped_reward -= float(self.config.switch_penalty)
        if option.name == "idle":
            shaped_reward -= float(self.config.inactivity_penalty)

        info["meta_reward"] = shaped_reward
        info["raw_reward"] = cumulative_reward
        info["option_elapsed"] = self.option_elapsed
        info["active_option_index"] = self.current_option_idx
        next_obs = self._build_meta_state(self.last_env_obs)
        return next_obs, float(shaped_reward), bool(done), info

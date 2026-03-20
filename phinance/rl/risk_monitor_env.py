"""RL environment for adaptive portfolio risk management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phinance.portfolio.simulator import PortfolioSimulator
from phinance.risk.metrics import compute_beta, compute_var_95

try:  # pragma: no cover
    from areal.rl import BaseEnv  # type: ignore
except Exception:  # pragma: no cover
    class BaseEnv:  # noqa: D401
        """Compatibility fallback when AReaL is unavailable."""


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
            self.n = n


STATE_FEATURES = [
    "drawdown",
    "var_95",
    "beta",
    "greeks_delta",
    "greeks_gamma",
    "greeks_vega",
    "regime_bull",
    "regime_bear",
    "regime_sideways",
    "volatility_20",
    "correlation",
    "leverage_ratio",
    "rebalance_age",
    "exploration_count",
]

RISK_PROFILES: List[Dict[str, float]] = [
    {"name": "ultra_conservative", "max_position_size": 0.05, "stop_loss": 0.02, "var_limit": 0.01, "hedge_ratio": 0.0},
    {"name": "conservative", "max_position_size": 0.10, "stop_loss": 0.05, "var_limit": 0.02, "hedge_ratio": 0.0},
    {"name": "moderate", "max_position_size": 0.20, "stop_loss": 0.08, "var_limit": 0.04, "hedge_ratio": 0.0},
    {"name": "aggressive", "max_position_size": 0.40, "stop_loss": 0.12, "var_limit": 0.08, "hedge_ratio": 0.0},
    {"name": "aggressive_hedge", "max_position_size": 0.40, "stop_loss": 0.12, "var_limit": 0.08, "hedge_ratio": 0.20},
]


@dataclass
class RiskMonitorEnvConfig:
    episode_length: int = 252
    step_days: int = 5
    catastrophic_drawdown: float = 0.5
    penalty_coefficient: float = 2.0
    initial_capital: float = 100000.0
    seed: int = 19
    data_points: int = 2520


class RiskMonitorEnv(BaseEnv):
    """Discrete-action environment mapping actions to portfolio risk profiles."""

    observation_space = Box(low=-1.0, high=1.0, shape=(len(STATE_FEATURES),), dtype=np.float32)

    def __init__(self, data: Optional[pd.DataFrame] = None, config: Optional[RiskMonitorEnvConfig] = None) -> None:
        self.config = config or RiskMonitorEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.data = self._prepare_data(data)
        self.returns = self.data["close"].pct_change().fillna(0.0).to_numpy(dtype=np.float64)
        self.vol_20 = pd.Series(self.returns).rolling(20).std().fillna(0.0).to_numpy(dtype=np.float64)
        self.action_space = Discrete(len(RISK_PROFILES))
        self.simulator = PortfolioSimulator(initial_capital=self.config.initial_capital)

        self.current_step = 0
        self.day_index = 20
        self.last_reward = 0.0
        self.last_rebalance_step = 0

    def _prepare_data(self, provided: Optional[pd.DataFrame]) -> pd.DataFrame:
        if provided is not None:
            df = provided.copy()
        else:
            idx = pd.date_range("2014-01-01", periods=self.config.data_points, freq="D")
            walk = 300 + np.cumsum(self.rng.normal(0.03, 1.5, size=len(idx)))
            df = pd.DataFrame({"open": walk, "high": walk * 1.01, "low": walk * 0.99, "close": walk, "volume": 1_000_000}, index=idx)

        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"RiskMonitorEnv data missing required columns: {sorted(missing)}")
        return df.reset_index(drop=True)

    def _regime_vector(self) -> np.ndarray:
        if self.day_index < 30:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        lookback = self.returns[max(0, self.day_index - 20): self.day_index]
        trend = float(np.mean(lookback))
        if trend > 0.002:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if trend < -0.002:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _sharpe(self) -> float:
        rets = np.asarray(self.simulator.return_history, dtype=np.float64)
        if rets.size < 2:
            return 0.0
        std = float(np.std(rets))
        if std <= 1e-12:
            return 0.0
        return float(np.mean(rets) / std * np.sqrt(252.0))

    def _build_state(self) -> np.ndarray:
        history = np.asarray(self.simulator.return_history[-60:], dtype=np.float64)
        benchmark = self.returns[max(0, self.day_index - history.size): self.day_index]
        var_95 = compute_var_95(history)
        beta = compute_beta(history, benchmark)
        regime = self._regime_vector()
        vol = float(np.clip(self.vol_20[min(self.day_index, len(self.vol_20) - 1)] * 40.0, 0.0, 1.0))

        state = np.array(
            [
                np.clip(self.simulator.drawdown, 0.0, 1.0),
                np.clip(var_95 * 10.0, 0.0, 1.0),
                np.clip(beta / 2.0, -1.0, 1.0),
                np.clip(self.simulator.leverage_ratio, -1.0, 1.0),
                0.0,
                0.0,
                *regime,
                vol,
                1.0,
                np.clip(self.simulator.leverage_ratio, 0.0, 1.0),
                np.clip((self.current_step - self.last_rebalance_step) / max(self.config.episode_length, 1), 0.0, 1.0),
                np.clip(self.current_step / max(self.config.episode_length, 1), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return state

    def _reward(self) -> float:
        sharpe = self._sharpe()
        max_dd = max((1.0 - (value / max(self.simulator.peak_value, 1e-9)) for value in self.simulator.value_history), default=0.0)
        return float(sharpe - self.config.penalty_coefficient * (max_dd ** 2))

    def reset(self) -> np.ndarray:
        self.simulator.reset()
        self.current_step = 0
        self.day_index = 20
        self.last_reward = 0.0
        self.last_rebalance_step = 0
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.current_step >= self.config.episode_length:
            raise RuntimeError("Episode already done. Call reset() before step().")

        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(RISK_PROFILES):
            raise ValueError(f"Action out of bounds: {action_idx}")

        profile = RISK_PROFILES[action_idx]
        self.simulator.set_risk_profile(profile["max_position_size"], profile["hedge_ratio"])
        self.last_rebalance_step = self.current_step

        limit_breach = False
        for _ in range(self.config.step_days):
            if self.day_index >= len(self.returns):
                break
            ret = float(self.returns[self.day_index])
            stats = self.simulator.step_day(ret, stop_loss=float(profile["stop_loss"]))
            if stats["drawdown"] >= self.config.catastrophic_drawdown:
                limit_breach = True
                break
            self.day_index += 1

        current_reward = self._reward()
        reward = float(current_reward - self.last_reward)
        self.last_reward = current_reward

        self.current_step += 1
        done = self.current_step >= self.config.episode_length or limit_breach
        if done and limit_breach:
            reward -= 5.0

        next_state = self._build_state() if not done else np.zeros(len(STATE_FEATURES), dtype=np.float32)
        info = {
            "profile": profile,
            "portfolio_value": self.simulator.total_value,
            "drawdown": self.simulator.drawdown,
            "sharpe": self._sharpe(),
            "catastrophic": limit_breach,
        }
        return next_state, reward, done, info

"""RL environment for strategy R&D template selection.

The environment is designed for discrete strategy selection where each action maps
into a precomputed strategy template + reward lookup.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from areal.rl import BaseEnv  # type: ignore
except Exception:  # pragma: no cover
    class BaseEnv:  # noqa: D401
        """Compatibility fallback when AReaL is unavailable."""


try:  # pragma: no cover - optional dependency
    from gymnasium.spaces import Box, Discrete
except Exception:  # pragma: no cover
    class Box:  # type: ignore[override]
        """Minimal Box replacement for local tests when gymnasium is absent."""

        def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class Discrete:  # type: ignore[override]
        """Minimal Discrete replacement for local tests when gymnasium is absent."""

        def __init__(self, n: int) -> None:
            self.n = n


STATE_FEATURES = [
    "regime_bull",
    "regime_bear",
    "regime_sideways",
    "volatility_20",
    "recent_performance",
    "exploration_count",
    "best_sharpe_so_far",
]


@dataclass
class StrategyRDEnvConfig:
    """Configuration for Strategy R&D environment."""

    episode_length: int = 10
    seed: int = 11
    data_points: int = 756
    initial_cash: float = 10000.0
    regime_lookback: int = 20
    indicator_catalog: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: {
            "rsi": {"period": [7, 14, 21], "oversold": [25, 30, 35], "overbought": [65, 70, 75]},
            "dual_sma": {"fast": [5, 10, 20], "slow": [30, 50, 100]},
            "bollinger": {"window": [10, 20, 30], "num_std": [1.5, 2.0, 2.5]},
        }
    )


class StrategyRDEnv(BaseEnv):
    """Discrete-action RL environment for strategy template exploration."""

    observation_space = Box(low=-1.0, high=1.0, shape=(len(STATE_FEATURES),), dtype=np.float32)

    def __init__(self, data: Optional[pd.DataFrame] = None, config: Optional[StrategyRDEnvConfig] = None) -> None:
        self.config = config or StrategyRDEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.data = self._prepare_data(data)

        self.returns = self.data["close"].pct_change().fillna(0.0)
        self.volatility_series = self.returns.rolling(20).std().fillna(0.0)

        self.templates = self._build_templates(self.config.indicator_catalog)
        self.template_rewards = self._precompute_template_rewards()
        self.benchmark_sharpe = self._compute_sharpe(self.returns)

        self.action_space = Discrete(len(self.templates))
        self.current_step = 0
        self.tried_actions: set[int] = set()
        self.best_sharpe = -np.inf
        self.last_reward = 0.0

    def _prepare_data(self, provided: Optional[pd.DataFrame]) -> pd.DataFrame:
        if provided is not None:
            df = provided.copy()
        else:
            idx = pd.date_range("2021-01-01", periods=self.config.data_points, freq="D")
            walk = 100 + np.cumsum(self.rng.normal(0.0, 1.0, size=len(idx)))
            df = pd.DataFrame(
                {
                    "open": walk,
                    "high": walk * (1.0 + 0.01),
                    "low": walk * (1.0 - 0.01),
                    "close": walk + self.rng.normal(0.0, 0.5, size=len(idx)),
                    "volume": self.rng.integers(1_000_000, 5_000_000, size=len(idx)),
                },
                index=idx,
            )

        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"StrategyRDEnv data missing required columns: {sorted(missing)}")

        return df.reset_index(drop=True)

    def _build_templates(self, catalog: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, Any]]:
        templates: List[Dict[str, Any]] = []
        for indicator, params in catalog.items():
            keys = list(params.keys())
            combinations = itertools.product(*(params[k] for k in keys))
            for combo in combinations:
                values = dict(zip(keys, combo))
                templates.append({"name": indicator, "params": values})
        if not templates:
            raise ValueError("Strategy template catalog produced zero templates")
        logger.info("Built %d strategy templates", len(templates))
        return templates

    def _compute_sharpe(self, returns: pd.Series) -> float:
        std = float(returns.std())
        if std <= 1e-12:
            return 0.0
        return float((returns.mean() / std) * np.sqrt(252.0))

    def _signals_for_template(self, template: Dict[str, Any]) -> pd.Series:
        close = self.data["close"]
        name = template["name"]
        params = template["params"]

        if name == "rsi":
            period = int(params["period"])
            oversold = float(params["oversold"])
            overbought = float(params["overbought"])
            delta = close.diff().fillna(0.0)
            gains = delta.clip(lower=0.0).rolling(period).mean()
            losses = (-delta.clip(upper=0.0)).rolling(period).mean().replace(0.0, np.nan)
            rs = gains / losses
            rsi = 100.0 - (100.0 / (1.0 + rs))
            signal = pd.Series(0.0, index=close.index)
            signal = signal.mask(rsi < oversold, 1.0)
            signal = signal.mask(rsi > overbought, -1.0)
            return signal.fillna(0.0)

        if name == "dual_sma":
            fast = int(params["fast"])
            slow = int(params["slow"])
            if fast >= slow:
                return pd.Series(0.0, index=close.index)
            fast_sma = close.rolling(fast).mean()
            slow_sma = close.rolling(slow).mean()
            return pd.Series(np.where(fast_sma > slow_sma, 1.0, -1.0), index=close.index).fillna(0.0)

        if name == "bollinger":
            window = int(params["window"])
            num_std = float(params["num_std"])
            mean = close.rolling(window).mean()
            std = close.rolling(window).std().fillna(0.0)
            upper = mean + num_std * std
            lower = mean - num_std * std
            signal = pd.Series(0.0, index=close.index)
            signal = signal.mask(close < lower, 1.0)
            signal = signal.mask(close > upper, -1.0)
            return signal.fillna(0.0)

        return pd.Series(0.0, index=close.index)

    def _precompute_template_rewards(self) -> np.ndarray:
        rewards: List[float] = []
        benchmark = self._compute_sharpe(self.returns)

        for template in self.templates:
            try:
                signal = self._signals_for_template(template)
                strategy_returns = signal.shift(1).fillna(0.0) * self.returns
                strategy_sharpe = self._compute_sharpe(strategy_returns)
                rewards.append(strategy_sharpe - benchmark)
            except Exception:
                rewards.append(-1.0)

        return np.asarray(rewards, dtype=np.float32)

    def _regime_vector(self) -> np.ndarray:
        trend = float(self.data["close"].pct_change(self.config.regime_lookback).iloc[-1])
        if trend > 0.02:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if trend < -0.02:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _build_state(self) -> np.ndarray:
        regime = self._regime_vector()
        volatility = float(np.clip(self.volatility_series.iloc[-1] * 50.0, 0.0, 1.0))
        recent_perf = float(np.clip(self.last_reward, -1.0, 1.0))
        explore = float(np.clip(self.current_step / max(self.config.episode_length, 1), 0.0, 1.0))
        best_sharpe = 0.0 if not np.isfinite(self.best_sharpe) else float(np.clip(self.best_sharpe / 3.0, -1.0, 1.0))

        return np.array([*regime, volatility, recent_perf, explore, best_sharpe], dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.tried_actions.clear()
        self.best_sharpe = -np.inf
        self.last_reward = 0.0
        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.current_step >= self.config.episode_length:
            raise RuntimeError("Episode already done. Call reset() before step().")

        action_idx = int(action)
        if action_idx < 0 or action_idx >= len(self.templates):
            raise ValueError(f"Action out of bounds: {action_idx}")

        duplicate_penalty = 0.0
        if action_idx in self.tried_actions:
            duplicate_penalty = -0.05
        self.tried_actions.add(action_idx)

        reward = float(self.template_rewards[action_idx] + duplicate_penalty)
        strategy_sharpe = reward + self.benchmark_sharpe
        self.best_sharpe = max(self.best_sharpe, strategy_sharpe)
        self.last_reward = reward

        self.current_step += 1
        done = self.current_step >= self.config.episode_length
        next_state = self._build_state() if not done else np.zeros(len(STATE_FEATURES), dtype=np.float32)
        info = {
            "template": self.templates[action_idx],
            "template_index": action_idx,
            "strategy_sharpe": float(strategy_sharpe),
            "benchmark_sharpe": float(self.benchmark_sharpe),
            "duplicate_penalty": float(duplicate_penalty),
        }
        return next_state, reward, done, info

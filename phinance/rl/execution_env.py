"""RL environment for optimal execution tasks.

The environment is intentionally lightweight and can run either with AReaL's
``BaseEnv`` interface (when installed) or with a small local compatibility
fallback used by tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from phinance.data.cache import DataCache
from phinance.rl.training_utils import python_step_kernel, state_cache, step_kernel

logger = logging.getLogger(__name__)

try:  # pragma: no cover - covered via compatibility fallback tests
    from areal.rl import BaseEnv  # type: ignore
except Exception:  # pragma: no cover
    class BaseEnv:  # noqa: D401
        """Compatibility fallback when AReaL is unavailable."""


try:  # pragma: no cover - optional dependency
    from gymnasium.spaces import Box
except Exception:  # pragma: no cover
    class Box:  # type: ignore[override]
        """Minimal Box replacement for local tests when gymnasium is absent."""

        def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype


STATE_FEATURES = [
    "remaining_shares",
    "time_remaining",
    "bid_volume",
    "ask_volume",
    "spread",
    "volatility",
    "imbalance",
    "mid_price_return",
    "liquidity_score",
]


@dataclass
class ExecutionEnvConfig:
    """Configuration parameters for the execution environment."""

    symbol: str = "SPY"
    timeframe: str = "1m"
    vendor: str = "yfinance"
    episode_length: int = 30
    participation_cap: float = 0.25
    impact_coefficient: float = 0.1
    urgency_cross_spread: float = 1.0
    seed: int = 7
    enable_numba: bool = False
    enable_state_cache: bool = False


class ExecutionEnv(BaseEnv):
    """Environment for training an execution policy."""

    observation_space = Box(low=0.0, high=1.0, shape=(len(STATE_FEATURES),), dtype=np.float32)
    action_space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        config: Optional[ExecutionEnvConfig] = None,
        cache: Optional[DataCache] = None,
    ) -> None:
        self.config = config or ExecutionEnvConfig()
        self.cache = cache or DataCache()
        self.rng = np.random.default_rng(self.config.seed)
        self.data = self._prepare_data(data)
        self.current_step = 0
        self.start_idx = 0
        self.end_idx = self.config.episode_length
        self.total_shares = 0.0
        self.remaining_shares = 0.0
        self.arrival_price = 0.0
        self.done = False
        self._adv_series = self.data["volume"].rolling(390).mean().to_numpy(dtype=np.float64)
        self._obs_builder = self._build_obs_cached if self.config.enable_state_cache else self._build_obs_uncached
        self._step_kernel = step_kernel if self.config.enable_numba else python_step_kernel

    def _prepare_data(self, provided: Optional[pd.DataFrame]) -> pd.DataFrame:
        if provided is not None:
            return self._enrich_frame(provided)

        datasets = self.cache.list_datasets()
        for item in datasets:
            if (
                item.get("symbol") == self.config.symbol
                and item.get("timeframe") == self.config.timeframe
                and item.get("vendor") == self.config.vendor
            ):
                frame = pd.read_parquet(Path(item["path"]))
                logger.info("Loaded cached data %s", item["path"])
                return self._enrich_frame(frame)

        logger.warning("No cached dataset found for %s/%s, creating synthetic data", self.config.symbol, self.config.timeframe)
        idx = pd.date_range("2024-01-01", periods=1000, freq="min")
        walk = 100 + np.cumsum(self.rng.normal(0.0, 0.05, size=len(idx)))
        synthetic = pd.DataFrame(
            {
                "open": walk,
                "high": walk * (1.0 + 0.0008),
                "low": walk * (1.0 - 0.0008),
                "close": walk + self.rng.normal(0, 0.02, size=len(idx)),
                "volume": self.rng.integers(8000, 25000, size=len(idx)),
            },
            index=idx,
        )
        return self._enrich_frame(synthetic)

    def _enrich_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"ExecutionEnv data missing required columns: {sorted(missing)}")

        df["mid"] = (df["high"] + df["low"]) / 2.0
        df["spread"] = (df["high"] - df["low"]).clip(lower=1e-6)
        df["return_1"] = df["mid"].pct_change().fillna(0.0)
        df["volatility_5"] = df["return_1"].rolling(5).std().fillna(0.0)
        df["bid_volume"] = (df["volume"] * 0.5 * (1 + np.tanh(-df["return_1"] * 50))).clip(lower=1.0)
        df["ask_volume"] = (df["volume"] - df["bid_volume"]).clip(lower=1.0)
        volume_rank = (df["volume"] - df["volume"].rolling(50).mean().fillna(df["volume"].mean()))
        volume_scale = (df["volume"].rolling(50).std().fillna(df["volume"].std() or 1.0)).replace(0, 1.0)
        spread_baseline = df["spread"].rolling(50).mean().fillna(df["spread"].mean())
        df["liquidity_score"] = np.tanh(volume_rank / volume_scale) - np.tanh((df["spread"] / spread_baseline) - 1.0)
        return df.dropna().reset_index(drop=True)

    def reset(self) -> np.ndarray:
        max_start = len(self.data) - self.config.episode_length - 1
        if max_start <= 1:
            raise ValueError("Dataset too small for configured episode length")
        self.start_idx = int(self.rng.integers(0, max_start))
        self.end_idx = self.start_idx + self.config.episode_length
        self.current_step = self.start_idx

        adv_raw = float(self._adv_series[self.current_step])
        adv = adv_raw if np.isfinite(adv_raw) and adv_raw > 0 else float(self.data["volume"].mean())
        self.total_shares = max(100.0, adv * 0.01)
        self.remaining_shares = self.total_shares
        self.arrival_price = float(self.data.iloc[self.current_step]["mid"])
        self.done = False
        return self._get_observation()

    @state_cache(maxsize=16384)
    def _build_obs_cached(self, state: np.ndarray) -> np.ndarray:
        return state

    def _build_obs_uncached(self, state: np.ndarray) -> np.ndarray:
        return state

    def _get_observation(self) -> np.ndarray:
        row = self.data.iloc[self.current_step]
        state = np.array(
            [
                np.clip(self.remaining_shares / max(self.total_shares, 1.0), 0.0, 1.0),
                np.clip((self.end_idx - self.current_step) / max(self.config.episode_length, 1), 0.0, 1.0),
                np.clip(row["bid_volume"] / max(row["volume"], 1.0), 0.0, 1.0),
                np.clip(row["ask_volume"] / max(row["volume"], 1.0), 0.0, 1.0),
                np.clip(row["spread"] / max(row["mid"], 1e-6) * 100.0, 0.0, 1.0),
                np.clip(row["volatility_5"] * 100.0, 0.0, 1.0),
                np.clip((row["bid_volume"] - row["ask_volume"]) / max(row["bid_volume"] + row["ask_volume"], 1.0) * 0.5 + 0.5, 0.0, 1.0),
                np.clip(row["return_1"] * 20.0 + 0.5, 0.0, 1.0),
                np.clip((row["liquidity_score"] + 2.0) / 4.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return self._obs_builder(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already done. Call reset() before step().")

        action = np.asarray(action, dtype=np.float32)
        fraction = float(np.clip(action[0], 0.0, 1.0))
        urgency = float(np.clip(action[1], 0.0, 1.0))

        row = self.data.iloc[self.current_step]
        adv_raw = float(self._adv_series[self.current_step])
        adv = adv_raw if np.isfinite(adv_raw) and adv_raw > 0 else float(self.data["volume"].mean())

        executed_shares, execution_price, slippage, reward, _ = self._step_kernel(
            float(self.remaining_shares),
            fraction,
            float(row["volume"]),
            float(self.config.participation_cap),
            float(row["spread"]),
            float(row["mid"]),
            float(self.arrival_price),
            float(self.config.impact_coefficient),
            urgency,
            float(self.config.urgency_cross_spread),
            float(adv),
            float(self.total_shares),
        )

        self.remaining_shares -= float(executed_shares)
        self.current_step += 1

        forced_liquidation_cost = 0.0
        if self.current_step >= self.end_idx and self.remaining_shares > 0:
            forced_execution = float(self.data.iloc[self.current_step - 1]["mid"] + row["spread"])
            forced_slippage = (forced_execution - self.arrival_price) / max(self.arrival_price, 1e-6)
            forced_liquidation_cost = forced_slippage * (self.remaining_shares / max(self.total_shares, 1.0))
            reward -= forced_liquidation_cost
            executed_shares += self.remaining_shares
            self.remaining_shares = 0.0

        self.done = self.remaining_shares <= 0.0 or self.current_step >= self.end_idx
        next_obs = self._get_observation() if not self.done else np.zeros(len(STATE_FEATURES), dtype=np.float32)
        info = {
            "executed_shares": float(executed_shares),
            "remaining_shares": float(self.remaining_shares),
            "execution_price": float(execution_price),
            "slippage": float(slippage),
            "forced_liquidation_cost": float(forced_liquidation_cost),
            "arrival_price": float(self.arrival_price),
        }
        return next_obs, float(reward), self.done, info

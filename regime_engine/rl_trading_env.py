"""
Reinforcement Learning Trading Environment
-------------------------------------------
OpenAI Gymnasium environment for training RL agents to trade using
regime features from the Phi-nance FeatureEngine.

Action space:  Discrete(4) → [HOLD=0, BUY=1, SELL=2, EXIT=3]
Observation:   regime features (from FeatureEngine) + portfolio state

Compatible with Stable-Baselines3 (PPO, A2C, SAC, etc.)

Usage:
    See `train_rl_agent.py` for a full training example.

Requirements:
    pip install stable-baselines3 gymnasium
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from regime_engine.features import FeatureEngine


if not _GYM_AVAILABLE:
    # Provide a stub so the module can be imported without gymnasium installed
    class TradingEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gymnasium is required for TradingEnv. "
                "Run: pip install stable-baselines3 gymnasium"
            )
else:
    class TradingEnv(gym.Env):
        """
        Single-asset RL trading environment.

        Parameters
        ----------
        data_df : pd.DataFrame
            OHLCV DataFrame **with** pre-computed regime feature columns
            (output of FeatureEngine.compute() joined to OHLCV).
        initial_cash : float
            Starting portfolio value in dollars.
        feature_cols : list[str] | None
            Which columns to include in observation.
            Defaults to FeatureEngine.FEATURE_COLS.
        """

        metadata = {"render_modes": ["human"]}

        # Actions
        HOLD = 0
        BUY  = 1
        SELL = 2
        EXIT = 3

        def __init__(
            self,
            data_df: pd.DataFrame,
            initial_cash: float = 100_000.0,
            feature_cols: list[str] | None = None,
        ) -> None:
            super().__init__()
            self.data         = data_df.reset_index(drop=True)
            self.initial_cash = initial_cash
            self.feature_cols = feature_cols or FeatureEngine.FEATURE_COLS

            # Validate columns exist (fill missing with 0)
            for col in self.feature_cols:
                if col not in self.data.columns:
                    self.data[col] = 0.0

            # Require close price
            if "close" not in self.data.columns:
                raise ValueError("data_df must have a 'close' column.")

            self.action_space = spaces.Discrete(4)

            # Observation: features + [cash_ratio, position_norm, price_norm]
            n_obs = len(self.feature_cols) + 3
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
            )

            # State
            self.cash         = initial_cash
            self.position     = 0.0          # number of shares held
            self.current_step = 0
            self.max_steps    = len(self.data) - 2  # leave 1 for reward lookahead

        # ------------------------------------------------------------------
        # Gymnasium API
        # ------------------------------------------------------------------

        def reset(  # type: ignore[override]
            self,
            *,
            seed: int | None = None,
            options: dict | None = None,
        ) -> tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self.cash         = self.initial_cash
            self.position     = 0.0
            self.current_step = 0
            return self._get_obs(), {}

        def step(
            self, action: int
        ) -> tuple[np.ndarray, float, bool, bool, dict]:
            row          = self.data.iloc[self.current_step]
            next_row     = self.data.iloc[self.current_step + 1]
            current_price = float(row["close"])
            next_price    = float(next_row["close"])

            # Execute action
            if action == self.BUY and self.cash > 0:
                shares = self.cash / current_price
                self.position += shares
                self.cash = 0.0
            elif action in (self.SELL, self.EXIT) and self.position > 0:
                self.cash    += self.position * current_price
                self.position = 0.0
            # HOLD → do nothing

            # Portfolio value after price moves
            portfolio_value = self.cash + self.position * next_price
            prev_portfolio  = self.cash + self.position * current_price
            reward = float((portfolio_value - prev_portfolio) / self.initial_cash)

            self.current_step += 1
            terminated = self.current_step >= self.max_steps
            truncated  = False

            info = {
                "portfolio_value": portfolio_value,
                "position": self.position,
                "cash": self.cash,
                "step": self.current_step,
            }
            return self._get_obs(), reward, terminated, truncated, info

        def render(self, mode: str = "human") -> None:
            row = self.data.iloc[self.current_step]
            pv  = self.cash + self.position * float(row["close"])
            print(
                f"Step {self.current_step:4d} | "
                f"Price ${float(row['close']):8.2f} | "
                f"Portfolio ${pv:10.2f} | "
                f"Cash ${self.cash:10.2f} | "
                f"Pos {self.position:.3f} shares"
            )

        # ------------------------------------------------------------------
        # Internal
        # ------------------------------------------------------------------

        def _get_obs(self) -> np.ndarray:
            row = self.data.iloc[self.current_step]
            features = row[self.feature_cols].values.astype(np.float32)
            np.nan_to_num(features, copy=False)

            price = float(row["close"])
            portfolio_value = self.cash + self.position * price

            extra = np.array([
                self.cash / self.initial_cash,             # cash ratio
                self.position * price / self.initial_cash, # position ratio
                price / (self.data["close"].mean() + 1e-8),# normalised price level
            ], dtype=np.float32)

            return np.concatenate([features, extra])

"""Execution agent with RL policy support and TWAP fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch

from phinance.ml.inference import TransformerFeatureExtractor
from phinance.rl.execution_env import STATE_FEATURES


@dataclass
class ExecutionDecision:
    """Output for one execution decision step."""

    shares_to_trade: float
    urgency: float


class _PolicyWrapper:
    """Tiny inference wrapper around a saved PyTorch policy checkpoint."""

    def __init__(self, policy_path: Path) -> None:
        payload = torch.load(policy_path, map_location="cpu")
        self.obs_dim = int(payload.get("obs_dim", len(STATE_FEATURES)))

        from phinance.rl.policy_networks import GaussianPolicy

        model = GaussianPolicy(
            obs_dim=self.obs_dim,
            action_dim=int(payload.get("action_dim", 2)),
            architecture=str(payload.get("architecture", "mlp")),
            sequence_length=int(payload.get("sequence_length", 16)),
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self.model = model

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mean, std = self.model(tensor)
            action = mean if deterministic else torch.normal(mean, std)
        return torch.clamp(action.squeeze(0), 0.0, 1.0).cpu().numpy()


def load_rl_policy(policy_path: str | Path) -> _PolicyWrapper:
    """Load an RL execution policy from disk."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Execution policy checkpoint not found: {path}")
    return _PolicyWrapper(path)


class ExecutionAgent:
    """Convert large parent orders into dynamic slices."""

    def __init__(
        self,
        use_rl: bool = True,
        policy_path: str = "models/execution_agent/latest.pt",
        use_transformer_embeddings: bool = False,
        transformer_model_path: str = "phinance/ml/checkpoints/transformer_latest.pt",
    ) -> None:
        self.policy: Optional[_PolicyWrapper] = None
        self.transformer: Optional[TransformerFeatureExtractor] = None
        if use_rl:
            try:
                self.policy = load_rl_policy(policy_path)
            except FileNotFoundError:
                self.policy = None
        if use_transformer_embeddings:
            try:
                self.transformer = TransformerFeatureExtractor(transformer_model_path)
            except FileNotFoundError:
                self.transformer = None

    def _build_state(self, order: Dict[str, Any], market_data: pd.DataFrame) -> np.ndarray:
        row = market_data.iloc[-1]
        total = float(order.get("total_shares", order.get("qty", 1.0)))
        remaining = float(order.get("remaining_shares", total))
        horizon = max(float(order.get("horizon_steps", 1.0)), 1.0)
        remaining_steps = float(order.get("remaining_steps", horizon))

        bid_volume = float(row.get("bid_volume", row.get("volume", 1.0) * 0.5))
        ask_volume = float(row.get("ask_volume", row.get("volume", 1.0) * 0.5))
        volume = max(float(row.get("volume", bid_volume + ask_volume)), 1.0)
        mid = float(row.get("mid", (row.get("high", row["close"]) + row.get("low", row["close"])) / 2.0))
        spread = float(row.get("spread", max(row.get("high", mid) - row.get("low", mid), 1e-6)))
        returns = market_data["close"].pct_change().fillna(0.0)
        vol = float(returns.tail(5).std() if len(returns) >= 5 else returns.std())
        ret = float(returns.iloc[-1]) if len(returns) else 0.0
        liq_raw = (volume / max(market_data["volume"].tail(50).mean(), 1.0)) - (spread / max(mid, 1e-6))

        state = np.array(
            [
                np.clip(remaining / max(total, 1.0), 0.0, 1.0),
                np.clip(remaining_steps / horizon, 0.0, 1.0),
                np.clip(bid_volume / volume, 0.0, 1.0),
                np.clip(ask_volume / volume, 0.0, 1.0),
                np.clip((spread / max(mid, 1e-6)) * 100.0, 0.0, 1.0),
                np.clip(vol * 100.0, 0.0, 1.0),
                np.clip((bid_volume - ask_volume) / max(bid_volume + ask_volume, 1.0) * 0.5 + 0.5, 0.0, 1.0),
                np.clip(ret * 20.0 + 0.5, 0.0, 1.0),
                np.clip((liq_raw + 1.0) / 2.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        if self.transformer is None:
            return state
        embedding = self.transformer.extract_embedding(market_data)
        return np.concatenate([state, embedding.astype(np.float32)])

    def _schedule_from_action(self, action: np.ndarray, order: Dict[str, Any]) -> ExecutionDecision:
        remaining = float(order.get("remaining_shares", order.get("qty", 0.0)))
        fraction = float(np.clip(action[0], 0.0, 1.0))
        urgency = float(np.clip(action[1], 0.0, 1.0))
        return ExecutionDecision(shares_to_trade=max(0.0, remaining * fraction), urgency=urgency)

    def _twap_execute(self, order: Dict[str, Any]) -> ExecutionDecision:
        remaining = float(order.get("remaining_shares", order.get("qty", 0.0)))
        steps = max(float(order.get("remaining_steps", order.get("horizon_steps", 1.0))), 1.0)
        return ExecutionDecision(shares_to_trade=remaining / steps, urgency=0.5)

    def execute_order(self, order: Dict[str, Any], market_data: pd.DataFrame) -> ExecutionDecision:
        """Return next execution slice from RL policy or TWAP fallback."""
        if self.policy is None:
            return self._twap_execute(order)

        state = self._build_state(order, market_data)
        action = self.policy.act(state, deterministic=True)
        return self._schedule_from_action(action, order)

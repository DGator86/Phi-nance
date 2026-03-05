"""Risk Monitor agent with RL policy support and profile fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from phinance.rl.risk_monitor_env import RISK_PROFILES, STATE_FEATURES


class _PolicyWrapper:
    def __init__(self, policy_path: Path) -> None:
        payload = torch.load(policy_path, map_location="cpu")
        self.obs_dim = int(payload.get("obs_dim", len(STATE_FEATURES)))
        self.n_actions = int(payload.get("n_actions", len(RISK_PROFILES)))
        self.profiles: List[Dict[str, float]] = payload.get("profiles", RISK_PROFILES)

        from scripts.train_risk_monitor_agent import RiskMonitorPolicy

        model = RiskMonitorPolicy(obs_dim=self.obs_dim, n_actions=self.n_actions)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self.model = model

    def act(self, state: np.ndarray, deterministic: bool = True) -> int:
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = self.model(tensor).squeeze(0)
            if deterministic:
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())


def load_risk_monitor_policy(policy_path: str | Path) -> _PolicyWrapper:
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Risk monitor policy checkpoint not found: {path}")
    return _PolicyWrapper(path)


class RiskMonitorAgent:
    """Emit dynamic risk limits from RL policy or moderate defaults."""

    def __init__(self, use_rl: bool = True, policy_path: str = "models/risk_monitor_agent/latest.pt") -> None:
        self.policy: Optional[_PolicyWrapper] = None
        if use_rl:
            try:
                self.policy = load_risk_monitor_policy(policy_path)
            except FileNotFoundError:
                self.policy = None

    def _build_state(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> np.ndarray:
        regime = str(market_data.get("regime", "sideways")).lower()
        regime_vec = {
            "bull": [1.0, 0.0, 0.0],
            "bear": [0.0, 1.0, 0.0],
            "sideways": [0.0, 0.0, 1.0],
        }.get(regime, [0.0, 0.0, 1.0])
        return np.array(
            [
                np.clip(float(portfolio_state.get("drawdown", 0.0)), 0.0, 1.0),
                np.clip(float(portfolio_state.get("var_95", 0.0)) * 10.0, 0.0, 1.0),
                np.clip(float(portfolio_state.get("beta", 0.0)) / 2.0, -1.0, 1.0),
                np.clip(float(portfolio_state.get("delta", 0.0)), -1.0, 1.0),
                np.clip(float(portfolio_state.get("gamma", 0.0)), -1.0, 1.0),
                np.clip(float(portfolio_state.get("vega", 0.0)), -1.0, 1.0),
                *regime_vec,
                np.clip(float(market_data.get("volatility", 0.0)), 0.0, 1.0),
                np.clip(float(portfolio_state.get("correlation", 0.0)), -1.0, 1.0),
                np.clip(float(portfolio_state.get("leverage_ratio", 0.0)), 0.0, 1.0),
                np.clip(float(portfolio_state.get("rebalance_age", 0.0)), 0.0, 1.0),
                np.clip(float(market_data.get("exploration_count", 0.0)), 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _risk_profile_from_action(self, action: int) -> Dict[str, float]:
        profile = dict(RISK_PROFILES[int(action) % len(RISK_PROFILES)])
        hedge_ratio = float(profile.get("hedge_ratio", 0.0))
        profile["hedge_action"] = "buy_put_protection" if hedge_ratio > 0 else "none"
        profile["hedge_template"] = {"type": "put", "moneyness": 0.98, "dte": 30, "ratio": hedge_ratio}
        return profile

    def get_risk_limits(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        if self.policy is None:
            return self._risk_profile_from_action(2)
        state = self._build_state(portfolio_state, market_data)
        action = self.policy.act(state, deterministic=True)
        return self._risk_profile_from_action(action)

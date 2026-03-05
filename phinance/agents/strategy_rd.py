"""Strategy R&D agent with RL policy support and random fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from phinance.rl.strategy_rd_env import STATE_FEATURES, StrategyRDEnv


class _PolicyWrapper:
    """Inference wrapper around saved strategy R&D policy."""

    def __init__(self, policy_path: Path) -> None:
        payload = torch.load(policy_path, map_location="cpu")
        self.obs_dim = int(payload.get("obs_dim", len(STATE_FEATURES)))
        self.n_actions = int(payload["n_actions"])
        self.templates: List[Dict[str, Any]] = payload.get("templates", [])

        from scripts.train_strategy_rd_agent import StrategyRDPolicy

        model = StrategyRDPolicy(obs_dim=self.obs_dim, n_actions=self.n_actions)
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


def load_strategy_rd_policy(policy_path: str | Path) -> _PolicyWrapper:
    """Load a strategy R&D RL policy checkpoint."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy R&D policy checkpoint not found: {path}")
    return _PolicyWrapper(path)


class StrategyRDAgent:
    """Suggest strategy templates from RL policy or random fallback."""

    def __init__(self, use_rl: bool = True, policy_path: str = "models/strategy_rd_agent/latest.pt") -> None:
        self.policy: Optional[_PolicyWrapper] = None
        self.template_library: List[Dict[str, Any]] = []
        if use_rl:
            try:
                self.policy = load_strategy_rd_policy(policy_path)
                self.template_library = self.policy.templates
            except FileNotFoundError:
                self.policy = None

        if not self.template_library:
            self.template_library = StrategyRDEnv().templates

    def _build_state(self, market_state: Dict[str, float]) -> np.ndarray:
        regime = market_state.get("regime", "sideways")
        if regime == "bull":
            regime_vector = [1.0, 0.0, 0.0]
        elif regime == "bear":
            regime_vector = [0.0, 1.0, 0.0]
        else:
            regime_vector = [0.0, 0.0, 1.0]

        volatility = float(np.clip(market_state.get("volatility", 0.0), 0.0, 1.0))
        recent_perf = float(np.clip(market_state.get("recent_performance", 0.0), -1.0, 1.0))
        exploration = float(np.clip(market_state.get("exploration_count", 0.0), 0.0, 1.0))
        best_sharpe = float(np.clip(market_state.get("best_sharpe", 0.0), -1.0, 1.0))

        return np.array([*regime_vector, volatility, recent_perf, exploration, best_sharpe], dtype=np.float32)

    def _template_from_index(self, idx: int) -> Dict[str, Any]:
        return self.template_library[int(idx) % len(self.template_library)]

    def _random_strategy(self) -> Dict[str, Any]:
        idx = int(np.random.randint(0, len(self.template_library)))
        return self._template_from_index(idx)

    def propose_strategy(self, market_state: Dict[str, float], deterministic: bool = True) -> Dict[str, Any]:
        """Propose one strategy template."""
        if self.policy is None:
            return self._random_strategy()

        state = self._build_state(market_state)
        action = self.policy.act(state, deterministic=deterministic)
        return self._template_from_index(action)

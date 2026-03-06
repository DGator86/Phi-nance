"""Strategy R&D agent with RL policy support and random fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from phinance.ml.inference import TransformerFeatureExtractor
from phinance.meta.vault_integration import load_discovered_templates
from phinance.rl.strategy_rd_env import STATE_FEATURES, StrategyRDEnv


class _PolicyWrapper:
    """Inference wrapper around saved strategy R&D policy."""

    def __init__(self, policy_path: Path) -> None:
        payload = torch.load(policy_path, map_location="cpu")
        self.obs_dim = int(payload.get("obs_dim", len(STATE_FEATURES)))
        self.n_actions = int(payload["n_actions"])
        self.templates: List[Dict[str, Any]] = payload.get("templates", [])

        from phinance.rl.policy_networks import CategoricalPolicy

        model = CategoricalPolicy(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            architecture=str(payload.get("architecture", "mlp")),
            sequence_length=int(payload.get("sequence_length", 16)),
        )
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



OPTION_STRATEGY_TEMPLATES = [
    {"name": "option_strategy", "params": {"template": "covered_call", "dte": 30, "moneyness": 1.02}},
    {"name": "option_strategy", "params": {"template": "protective_put", "dte": 30, "moneyness": 0.98}},
    {"name": "option_strategy", "params": {"template": "straddle", "dte": 21, "moneyness": 1.00}},
]

class StrategyRDAgent:
    """Suggest strategy templates from RL policy or random fallback."""

    def __init__(
        self,
        use_rl: bool = True,
        policy_path: str = "models/strategy_rd_agent/latest.pt",
        use_transformer_embeddings: bool = False,
        transformer_model_path: str = "phinance/ml/checkpoints/transformer_latest.pt",
        include_discovered: bool = True,
        strategy_vault_path: str = "data/strategy_vault.json",
    ) -> None:
        self.policy: Optional[_PolicyWrapper] = None
        self.transformer: Optional[TransformerFeatureExtractor] = None
        self.template_library: List[Dict[str, Any]] = []
        if use_rl:
            try:
                self.policy = load_strategy_rd_policy(policy_path)
                self.template_library = self.policy.templates
            except FileNotFoundError:
                self.policy = None

        if use_transformer_embeddings:
            try:
                self.transformer = TransformerFeatureExtractor(transformer_model_path)
            except FileNotFoundError:
                self.transformer = None

        if not self.template_library:
            self.template_library = StrategyRDEnv().templates

        # Phase 3: include predefined options templates in the action library.
        self.template_library.extend(OPTION_STRATEGY_TEMPLATES)
        if include_discovered:
            self.template_library.extend(load_discovered_templates(strategy_vault_path))

    def load_discovered_strategies(self, strategy_vault_path: str = "data/strategy_vault.json") -> int:
        """Load discovered strategies from vault and append to template library."""
        discovered = load_discovered_templates(strategy_vault_path)
        self.template_library.extend(discovered)
        return len(discovered)

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

        state = np.array([*regime_vector, volatility, recent_perf, exploration, best_sharpe], dtype=np.float32)
        if self.transformer is None:
            return state

        history = market_state.get("market_history")
        if history is None:
            return state
        embedding = self.transformer.extract_embedding(history)
        return np.concatenate([state, embedding.astype(np.float32)])

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

"""AI-driven blending powered by PhiAI helpers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BlendMethod
from .manual import WeightedSum


class AIDrivenBlend(BlendMethod):
    """Blend that can use pre-tuned weights or derive weights from PhiAI-style search."""

    def blend(self, signals: pd.DataFrame, **kwargs) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)

        tuned_weights: Optional[Dict[str, float]] = kwargs.get("weights")
        if kwargs.get("auto_tune", False) or not tuned_weights:
            tuned_weights = self._auto_tune_weights(signals)

        return WeightedSum().blend(signals, weights=tuned_weights)

    def get_params(self) -> dict:
        return {"weights": "dict[str, float]", "auto_tune": "bool"}

    def _auto_tune_weights(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Simple score-based weight search with optional PhiAI extension point."""
        # Extension point: importing keeps tight coupling optional.
        try:
            from phi.phiai.auto_tune import PhiAI  # noqa: F401
        except Exception:
            pass

        scores = signals.fillna(0.0).abs().mean(axis=0)
        if float(scores.sum()) <= 0:
            return {c: 1.0 / len(signals.columns) for c in signals.columns}
        weights = (scores / scores.sum()).to_dict()
        return {k: float(np.clip(v, 0.0, 1.0)) for k, v in weights.items()}

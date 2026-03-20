"""Placeholder Weights & Biases tracker.

W&B support is intentionally stubbed in this step; use backend `none` or `mlflow`.
"""

from __future__ import annotations

from phinance.experiment.tracker import WandBTrackerUnavailable


class WandBTracker(WandBTrackerUnavailable):
    """Compatibility alias for a future WandB tracker implementation."""


__all__ = ["WandBTracker"]

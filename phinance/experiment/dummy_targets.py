"""Dummy experiment targets for documentation/tests."""

from __future__ import annotations

from typing import Any


def gp_discovery_target(generations: int = 3, tracker: Any = None) -> dict[str, float]:
    best_score = 0.0
    for generation in range(generations):
        best_score += 1.0
        if tracker is not None:
            tracker.log_metrics({"best_score": best_score}, step=generation + 1)
    return {"best_score": best_score}

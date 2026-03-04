"""Blending facade."""

from __future__ import annotations

import importlib

import pandas as pd

from .registry import BLEND_METHODS


def blend_signals(
    signals: pd.DataFrame,
    weights: dict | None = None,
    method: str = "weighted_sum",
    regime_probs: pd.DataFrame | None = None,
    **kwargs,
) -> pd.Series:
    if signals.empty or signals.columns.empty:
        return pd.Series(dtype=float)

    method_path = BLEND_METHODS.get(method)
    if method_path is None:
        raise ValueError(f"Unknown blend method: {method}")

    module_path, class_name = method_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    blender_cls = getattr(module, class_name)
    return blender_cls().blend(signals, weights=weights, regime_probs=regime_probs, **kwargs)

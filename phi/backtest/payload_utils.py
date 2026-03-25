"""Pure helpers for backtest / blend payloads (no UI dependencies)."""

from __future__ import annotations

from typing import Any


def build_regime_boosts_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Map raw detector state keys to friendly regime labels for regime_weighted blending."""
    label_map: dict[str, str] = {str(k): str(v) for k, v in (payload.get("regime_label_map") or {}).items()}
    matrix = payload.get("regime_boost_matrix") or {}
    out: dict[str, dict[str, float]] = {}
    for raw_state, weights in matrix.items():
        if not isinstance(weights, dict):
            continue
        friendly = label_map.get(str(raw_state), str(raw_state))
        out[friendly] = {str(k): float(v) for k, v in weights.items()}
    return out

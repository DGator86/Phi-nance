"""Allocation strategies for portfolio backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class AllocationStrategy(ABC):
    """Base class for target portfolio weight generation."""

    @abstractmethod
    def allocate(
        self,
        capital: float,
        signals: dict[str, float],
        prices: dict[str, float],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Return target weights that sum to 1.0."""


def _normalize(weights: dict[str, float], fallback_symbols: list[str] | None = None) -> dict[str, float]:
    symbols = list(weights.keys()) or (fallback_symbols or [])
    clean = {s: max(0.0, float(weights.get(s, 0.0))) for s in symbols}
    total = float(sum(clean.values()))
    if total <= 0:
        if not symbols:
            return {}
        equal = 1.0 / len(symbols)
        return {s: equal for s in symbols}
    return {s: v / total for s, v in clean.items()}


class EqualWeightAllocation(AllocationStrategy):
    def allocate(self, capital: float, signals: dict[str, float], prices: dict[str, float], **kwargs: Any) -> dict[str, float]:
        symbols = [s for s, p in prices.items() if p and p > 0]
        if not symbols:
            return {}
        equal = 1.0 / len(symbols)
        return {s: equal for s in symbols}


class FixedWeightAllocation(AllocationStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = _normalize(weights)

    def allocate(self, capital: float, signals: dict[str, float], prices: dict[str, float], **kwargs: Any) -> dict[str, float]:
        symbols = [s for s in prices if prices[s] and prices[s] > 0]
        candidate = {s: self.weights.get(s, 0.0) for s in symbols}
        return _normalize(candidate, fallback_symbols=symbols)


class SignalWeightAllocation(AllocationStrategy):
    def allocate(self, capital: float, signals: dict[str, float], prices: dict[str, float], **kwargs: Any) -> dict[str, float]:
        weights = {s: abs(float(signals.get(s, 0.0))) for s in prices if prices[s] and prices[s] > 0}
        return _normalize(weights, fallback_symbols=list(weights.keys()))


class RiskParityAllocation(AllocationStrategy):
    def allocate(self, capital: float, signals: dict[str, float], prices: dict[str, float], **kwargs: Any) -> dict[str, float]:
        vol = kwargs.get("volatility", {}) or {}
        raw: dict[str, float] = {}
        for s, p in prices.items():
            if not p or p <= 0:
                continue
            sigma = float(vol.get(s, np.nan))
            if np.isnan(sigma) or sigma <= 0:
                sigma = 1.0
            raw[s] = 1.0 / sigma
        return _normalize(raw, fallback_symbols=list(raw.keys()))


def get_allocation_strategy(name: str, params: dict[str, Any] | None = None) -> AllocationStrategy:
    """Factory for configured allocation strategy instances."""
    params = params or {}
    key = str(name or "equal_weight").strip().lower()
    if key in {"equal", "equal_weight"}:
        return EqualWeightAllocation()
    if key in {"fixed", "fixed_weight"}:
        return FixedWeightAllocation(weights=params.get("weights", {}))
    if key in {"signal", "signal_weight", "signal_weighted"}:
        return SignalWeightAllocation()
    if key in {"risk_parity", "risk-parity"}:
        return RiskParityAllocation()
    raise ValueError(f"Unknown allocation strategy: {name}")


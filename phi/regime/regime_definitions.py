"""Canonical regime labels and helpers for options-aware regime composition."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from phi.regime.utils import extract_features

BASE_REGIMES: tuple[str, ...] = ("BULL", "BEAR", "RANGING")
VOLATILITY_REGIMES: tuple[str, ...] = ("LOW_VOL", "NORMAL_VOL", "HIGH_VOL")


@dataclass(frozen=True)
class DetailedRegime:
    """Composite regime with trend and volatility components."""

    trend: str
    volatility: str

    @property
    def label(self) -> str:
        return f"{self.trend}_{self.volatility}"


def _normalize_base_regime(label: str) -> str:
    text = str(label).upper()
    if any(token in text for token in ("BULL", "UP", "RISK_ON", "TREND_UP")):
        return "BULL"
    if any(token in text for token in ("BEAR", "DOWN", "RISK_OFF", "TREND_DN")):
        return "BEAR"
    if any(token in text for token in ("RANGE", "SIDE", "NEUTRAL", "CHOP")):
        return "RANGING"
    return "RANGING"


def infer_base_regime_from_prices(ohlcv: pd.DataFrame, short_window: int = 20, long_window: int = 100) -> str:
    """Infer a coarse trend regime from moving-average direction."""
    if short_window < 2 or long_window <= short_window:
        raise ValueError("Expected long_window > short_window >= 2")

    close = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["Close"]
    short_ma = close.rolling(short_window, min_periods=short_window).mean().iloc[-1]
    long_ma = close.rolling(long_window, min_periods=long_window).mean().iloc[-1]

    if np.isnan(short_ma) or np.isnan(long_ma):
        return "RANGING"
    if short_ma > long_ma:
        return "BULL"
    if short_ma < long_ma:
        return "BEAR"
    return "RANGING"


def infer_volatility_regime(
    ohlcv: pd.DataFrame,
    window: int = 20,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
) -> str:
    """Classify volatility using rolling realised volatility quantiles."""
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("Expected 0 <= low_quantile < high_quantile <= 1")

    features = extract_features(ohlcv, window=window)
    vol = features["rolling_vol"].dropna()
    if vol.empty:
        return "NORMAL_VOL"

    current = float(vol.iloc[-1])
    low_cut = float(vol.quantile(low_quantile))
    high_cut = float(vol.quantile(high_quantile))

    if current <= low_cut:
        return "LOW_VOL"
    if current >= high_cut:
        return "HIGH_VOL"
    return "NORMAL_VOL"


def compose_detailed_regime(base_regime: str, volatility_regime: str) -> DetailedRegime:
    """Build a validated composite regime object from trend and volatility labels."""
    trend = _normalize_base_regime(base_regime)
    vol = volatility_regime.upper()
    if vol not in VOLATILITY_REGIMES:
        raise ValueError(f"Unknown volatility regime '{volatility_regime}'")
    return DetailedRegime(trend=trend, volatility=vol)


def normalise_regime_probabilities(regime_probs: dict[str, float]) -> dict[str, float]:
    """Convert arbitrary labels to BULL/BEAR/RANGING and renormalise."""
    collapsed = dict.fromkeys(BASE_REGIMES, 0.0)
    for label, prob in regime_probs.items():
        collapsed[_normalize_base_regime(label)] += float(prob)

    total = sum(collapsed.values())
    if total <= 0:
        return dict.fromkeys(BASE_REGIMES, 0.0)
    return {k: v / total for k, v in collapsed.items()}


def detailed_labels_for_probabilities(
    base_probs: dict[str, float],
    vol_regime: str,
    allowed_base_regimes: Iterable[str] = BASE_REGIMES,
) -> dict[str, float]:
    """Project trend probabilities into detailed labels with fixed vol qualifier."""
    vol = vol_regime.upper()
    if vol not in VOLATILITY_REGIMES:
        raise ValueError(f"Unknown volatility regime '{vol_regime}'")

    norm = normalise_regime_probabilities(base_probs)
    return {f"{base}_{vol}": float(norm.get(base, 0.0)) for base in allowed_base_regimes}

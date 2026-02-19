"""
Regime Probability Field Engine
================================

Probabilistic hierarchical regime engine and regime-conditioned indicator
projection system for 1-minute equities (OHLCV only).

Architecture:  OHLCV → Features → Taxonomy (KPCOFGS) → Log-space Probability
               Field → 28 Species → 8 Collapsed Regimes → Validity-Gated
               Indicators → AR(1) Projection → Composite Score

Quick-start
-----------
>>> from regime_engine import UniverseScanner, RegimeEngine, load_config, simulate_ohlcv
>>> cfg = load_config()                       # loads config.yaml
>>> engine = RegimeEngine(cfg)
>>> ohlcv  = simulate_ohlcv(n_bars=1000)
>>> result = engine.run(ohlcv)
>>> print(result['regime_probs'].tail())
>>> print(result['mix'].tail())

Scanner mode
------------
>>> scanner = UniverseScanner()
>>> universe = {'AAPL': ohlcv_aapl, 'MSFT': ohlcv_msft, ...}
>>> ranked   = scanner.scan(universe)
"""

from .scanner import (
    RegimeEngine,
    UniverseScanner,
    TickerResult,
    load_config,
)
from .features import FeatureEngine
from .taxonomy_engine import TaxonomyEngine
from .probability_field import ProbabilityField
from .species import SPECIES_LIST, SPECIES_BY_ID, REGIME_BINS
from .indicator_library import INDICATOR_CLASSES, build_indicator
from .expert_registry import ExpertRegistry
from .projection_engine import ProjectionEngine
from .mixer import Mixer
from .data_fetcher import AlphaVantageFetcher, AlphaVantageMCP
from .live_scanner import LiveScanner
from .gamma_surface import GammaSurface
from .l2_feed import PolygonL2Client, PolygonRestClient


# ──────────────────────────────────────────────────────────────────────────────
# Simulation helper (used in demo and tests)
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd


def simulate_ohlcv(
    n_bars: int = 2000,
    seed: int = 42,
    regime_sequence: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLCV data with realistic regime transitions.

    Parameters
    ----------
    n_bars          : number of 1-minute bars
    seed            : random seed
    regime_sequence : if True, inject regime changes (trend → range → breakout)

    Returns
    -------
    pd.DataFrame with columns: open, high, low, close, volume
    """
    rng    = np.random.default_rng(seed)
    prices = np.empty(n_bars + 1)
    prices[0] = 100.0

    # Regime schedule (index boundaries)
    if regime_sequence and n_bars >= 600:
        regimes = [
            ("trend",   0,          n_bars // 4),
            ("range",   n_bars // 4, n_bars // 2),
            ("trend",   n_bars // 2, 3 * n_bars // 4),
            ("breakout", 3 * n_bars // 4, n_bars),
        ]
    else:
        regimes = [("trend", 0, n_bars)]

    returns = np.zeros(n_bars)
    vols    = np.zeros(n_bars)

    for rtype, start, end in regimes:
        n = end - start
        if rtype == "trend":
            drift = rng.choice([-1, 1]) * 0.0003
            vol   = 0.0012
            ret   = rng.normal(drift, vol, size=n)
        elif rtype == "range":
            # Mean-reverting: AR(1) with high mean-reversion
            ret   = np.zeros(n)
            ret[0] = rng.normal(0, 0.0010)
            for i in range(1, n):
                ret[i] = -0.35 * ret[i-1] + rng.normal(0, 0.0008)
            vol = np.abs(ret).mean() * 1.5
        elif rtype == "breakout":
            # Initial squeeze then explosive move
            half = n // 3
            ret1 = rng.normal(0, 0.0005, size=half)
            drift2 = rng.choice([-1, 1]) * 0.0006
            ret2 = rng.normal(drift2, 0.0018, size=n - half)
            ret  = np.concatenate([ret1, ret2])
            vol  = np.abs(ret).mean() * 1.5
        else:
            ret = rng.normal(0, 0.001, size=n)
            vol = 0.001

        returns[start:end] = ret
        vols[start:end]    = vol if np.isscalar(vol) else 0.001

    # Build price series
    for i in range(n_bars):
        prices[i + 1] = prices[i] * np.exp(returns[i])

    close = prices[1:]
    open_ = prices[:-1] * np.exp(rng.normal(0, 0.0002, size=n_bars))

    # High / low as spread around close
    spread = np.abs(rng.normal(0, 0.0008, size=n_bars))
    high   = np.maximum(open_, close) + spread * np.abs(rng.normal(1, 0.3, n_bars))
    low    = np.minimum(open_, close) - spread * np.abs(rng.normal(1, 0.3, n_bars))
    low    = np.minimum(low, np.minimum(open_, close))  # ensure low ≤ min(O,C)

    # Volume: positively correlated with absolute return
    base_vol = 50_000
    vol_mult = 1.0 + 3.0 * np.abs(returns) / (np.abs(returns).mean() + 1e-10)
    volume   = (base_vol * vol_mult * rng.lognormal(0, 0.4, n_bars)).astype(int)

    # Timestamps: simulate 1-minute bars starting at 09:30
    start_ts = pd.Timestamp("2024-01-02 09:30:00")
    idx = pd.date_range(start=start_ts, periods=n_bars, freq="1min")

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


__all__ = [
    "RegimeEngine",
    "UniverseScanner",
    "TickerResult",
    "load_config",
    "simulate_ohlcv",
    "FeatureEngine",
    "TaxonomyEngine",
    "ProbabilityField",
    "SPECIES_LIST",
    "SPECIES_BY_ID",
    "REGIME_BINS",
    "INDICATOR_CLASSES",
    "build_indicator",
    "ExpertRegistry",
    "ProjectionEngine",
    "Mixer",
    "AlphaVantageFetcher",
    "AlphaVantageMCP",
    "LiveScanner",
    "GammaSurface",
    "PolygonL2Client",
    "PolygonRestClient",
    # Interface 4 affinity blending utilities
    "compute_entropy_weighted_affinity",
    "entropy_certainty",
]

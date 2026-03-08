"""Training orchestration utilities for regime detectors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phi.config import settings
from phi.regime.base import RegimeDetector
from phi.regime.models.clustering import ClusteringRegimeDetector
from phi.regime.models.hmm import HMMRegimeDetector


def _build_model_filename(prefix: str, n_regimes: int, ohlcv: pd.DataFrame) -> str:
    start = str(pd.to_datetime(ohlcv.index.min()).date())
    end = str(pd.to_datetime(ohlcv.index.max()).date())
    return f"{prefix}_n{n_regimes}_{start}_{end}.pkl"


def train_regime_detector(
    ohlcv: pd.DataFrame,
    method: str = "hmm",
    n_regimes: int = 3,
    window: int = 20,
    save: bool = True,
) -> tuple[RegimeDetector, Path | None]:
    """Train requested detector and optionally persist it under ``settings.REGIME_MODELS_DIR``."""
    key = str(method).strip().lower()
    if key == "hmm":
        detector: RegimeDetector = HMMRegimeDetector(n_states=n_regimes).fit(ohlcv, window=window)
        filename = _build_model_filename("hmm", n_regimes, ohlcv)
    elif key in {"kmeans", "clustering"}:
        detector = ClusteringRegimeDetector(n_clusters=n_regimes, method="kmeans").fit(ohlcv, window=window)
        filename = _build_model_filename("clustering", n_regimes, ohlcv)
    else:
        raise ValueError(f"Unsupported regime detection method: {method}")

    if not save:
        return detector, None

    path = settings.REGIME_MODELS_DIR / filename
    detector.save(path)
    return detector, path

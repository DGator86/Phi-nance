"""Trainable market regime detection module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from phi.config import settings
from phi.regime.base import RegimeDetector
from phi.regime.models.clustering import ClusteringRegimeDetector
from phi.regime.models.hmm import HMMRegimeDetector
from phi.regime.train import train_regime_detector
from phi.regime.utils import extract_features


def _resolve_detector_class(detector_class: str) -> type[RegimeDetector]:
    """Map serialized detector class names to concrete implementations."""
    key = detector_class.strip().lower()
    if key == "hmmregimedetector":
        return HMMRegimeDetector
    if key == "clusteringregimedetector":
        return ClusteringRegimeDetector
    raise ValueError(f"Unsupported detector_class '{detector_class}'")


def list_saved_detectors(models_dir: Path | None = None) -> list[dict[str, Any]]:
    """List persisted regime detector models and metadata."""
    directory = Path(models_dir or settings.REGIME_MODELS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    detectors: list[dict[str, Any]] = []
    for model_path in sorted(directory.glob("*.pkl")):
        metadata: dict[str, Any] = {}
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        detectors.append(
            {
                "path": str(model_path),
                "name": model_path.stem,
                "metadata": metadata,
            }
        )
    return detectors


def load_detector(path: str | Path) -> RegimeDetector:
    """Load a saved regime detector from a ``.pkl`` artifact path."""
    model_path = Path(path)
    metadata_path = model_path.with_suffix(".json")
    detector_class = ""
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        detector_class = str(metadata.get("detector_class", ""))

    if not detector_class:
        # Filename fallback keeps old artifacts usable.
        stem = model_path.stem.lower()
        if stem.startswith("hmm"):
            detector_class = "HMMRegimeDetector"
        else:
            detector_class = "ClusteringRegimeDetector"

    cls = _resolve_detector_class(detector_class)
    return cls.load(model_path)


def predict_regimes(detector: RegimeDetector, ohlcv: pd.DataFrame) -> pd.Series:
    """Predict a regime series from OHLCV bars with a loaded detector."""
    return detector.predict(ohlcv)

__all__ = [
    "RegimeDetector",
    "HMMRegimeDetector",
    "ClusteringRegimeDetector",
    "extract_features",
    "train_regime_detector",
    "list_saved_detectors",
    "load_detector",
    "predict_regimes",
]

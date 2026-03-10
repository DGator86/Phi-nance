"""Trainable market regime detection module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from phi.config import settings
from phi.regime.base import RegimeDetector
from phi.regime.models.clustering import ClusteringRegimeDetector
from phi.regime.models.deep import DeepRegimeDetector
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
    if key == "deepregimedetector":
        return DeepRegimeDetector
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
        elif model_path.with_suffix(".scaler.pkl").exists() or "deep" in stem or "lstm" in stem or "transformer" in stem:
            detector_class = "DeepRegimeDetector"
        else:
            detector_class = "ClusteringRegimeDetector"

    cls = _resolve_detector_class(detector_class)
    return cls.load(model_path)


def predict_regimes(detector: RegimeDetector, ohlcv: pd.DataFrame) -> pd.Series:
    """Predict a regime series from OHLCV bars with a loaded detector."""
    return detector.predict(ohlcv)


def create_detector_from_params(detector_type: str, params: dict[str, Any]) -> RegimeDetector:
    """Create an unfitted regime detector from serialized optimization params."""
    key = detector_type.strip().lower()
    n_regimes = int(params.get("n_regimes", 3))
    random_state = int(params.get("random_state", 42))

    if key == "hmm":
        return HMMRegimeDetector(
            n_states=n_regimes,
            covariance_type=str(params.get("covariance_type", "diag")),
            random_state=random_state,
        )
    if key == "kmeans":
        return ClusteringRegimeDetector(
            n_clusters=n_regimes,
            method="kmeans",
            random_state=random_state,
        )
    if key == "gmm":
        return ClusteringRegimeDetector(
            n_clusters=n_regimes,
            method="gmm",
            random_state=random_state,
        )
    if key in {"deep_lstm", "lstm"}:
        return DeepRegimeDetector(model_type="lstm", **params)
    if key in {"deep_transformer", "transformer"}:
        return DeepRegimeDetector(model_type="transformer", **params)
    raise ValueError(f"Unknown detector type: {detector_type}")

__all__ = [
    "RegimeDetector",
    "HMMRegimeDetector",
    "ClusteringRegimeDetector",
    "DeepRegimeDetector",
    "extract_features",
    "train_regime_detector",
    "list_saved_detectors",
    "load_detector",
    "predict_regimes",
    "create_detector_from_params",
]

"""Load YAML training specs for regime detectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_regime_training_yaml(path: Path | str) -> dict[str, Any]:
    """Parse a regime training config file (see ``configs/ml/regime_train.example.yaml``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Regime training YAML must deserialize to a mapping at the root")
    return data

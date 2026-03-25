"""Versioned training manifests and QuantConnect-oriented artifact bundles."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1


def write_regime_training_manifest(
    model_path: Path,
    *,
    training_config: dict[str, Any],
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<stem>.manifest.json`` next to the ``.pkl`` for reproducibility."""
    model_path = Path(model_path)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name,
        "training_config": training_config,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if extra:
        manifest["extra"] = extra
    out = model_path.with_suffix(".manifest.json")
    out.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return out


def export_regime_bundle_for_quantconnect(
    model_pkl: Path | str,
    out_dir: Path | str,
    *,
    regimes_csv: Path | None = None,
) -> Path:
    """Copy detector files into a folder you can zip next to ``ohlcv.csv`` for QC research.

    Note: Lean/Python on QuantConnect cannot load arbitrary ``joblib`` sklearn pickles from
    Phi-nance in all cases; use exported **regime label CSV** (see
    ``scripts/export_regime_predictions_for_qc.py``) for custom data in QC, or re-train a
    QC-native model. This bundle is for provenance + local tooling.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    src = Path(model_pkl)
    shutil.copy2(src, root / "regime_detector.pkl")
    meta = src.with_suffix(".json")
    if meta.exists():
        shutil.copy2(meta, root / "regime_detector.metadata.json")
    man = src.with_suffix(".manifest.json")
    if man.exists():
        shutil.copy2(man, root / "regime_detector.manifest.json")
    if regimes_csv is not None and Path(regimes_csv).exists():
        shutil.copy2(regimes_csv, root / "regimes.csv")
    return root

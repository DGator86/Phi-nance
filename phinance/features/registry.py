"""Feature registry persistence for discovered feature definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class FeatureRegistry:
    """JSON-backed registry for autoencoder and GP features."""

    path: Path
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: str | Path) -> "FeatureRegistry":
        instance = cls(path=Path(path))
        instance.load()
        return instance

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.payload = {"autoencoder": None, "gp_features": []}
            return self.payload
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.payload.setdefault("autoencoder", None)
        self.payload.setdefault("gp_features", [])
        return self.payload

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")

    def set_autoencoder(self, checkpoint_path: str, latent_dim: int, metrics: dict[str, float] | None = None) -> None:
        self.payload["autoencoder"] = {
            "checkpoint_path": checkpoint_path,
            "latent_dim": int(latent_dim),
            "metrics": metrics or {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def add_gp_feature(self, expression: str, metrics: dict[str, float] | None = None) -> None:
        entry = {
            "expression": expression,
            "metrics": metrics or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        gp_features = self.payload.setdefault("gp_features", [])
        if not any(existing.get("expression") == expression for existing in gp_features):
            gp_features.append(entry)

    def top_gp_features(self, limit: int = 10) -> list[dict[str, Any]]:
        gp_features = self.payload.get("gp_features", [])
        ranked = sorted(gp_features, key=lambda x: float(x.get("metrics", {}).get("fitness", 0.0)), reverse=True)
        return ranked[:limit]

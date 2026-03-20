"""Orchestration pipeline for automated feature discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from phinance.features.autoencoder import AutoencoderConfig, MarketAutoencoder
from phinance.features.genetic_features import GPFeatureConfig, GPFeatureDiscovery
from phinance.features.registry import FeatureRegistry


@dataclass
class FeaturePipelineConfig:
    autoencoder: dict[str, Any]
    gp: dict[str, Any]
    registry_path: str
    checkpoint_path: str
    window: int = 32

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeaturePipelineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            autoencoder=raw.get("autoencoder", {}),
            gp=raw.get("gp", {}),
            registry_path=raw.get("registry_path", "phinance/features/feature_registry.json"),
            checkpoint_path=raw.get("checkpoint_path", "phinance/features/checkpoints/autoencoder_latest.npz"),
            window=int(raw.get("window", 32)),
        )


class FeatureDiscoveryPipeline:
    """Runs model training/evolution and updates the feature registry."""

    def __init__(self, config: FeaturePipelineConfig) -> None:
        self.config = config

    def _build_windows(self, frame: pd.DataFrame, window: int) -> np.ndarray:
        numeric = frame.select_dtypes(include=[np.number])
        if len(numeric) < window:
            raise ValueError(f"Need at least {window} rows, got {len(numeric)}")
        windows = []
        for i in range(window, len(numeric) + 1):
            windows.append(numeric.iloc[i - window : i].to_numpy(dtype=np.float32).flatten())
        return np.stack(windows)

    def run(self, frame: pd.DataFrame, enable_autoencoder: bool = True, enable_gp: bool = True) -> dict[str, Any]:
        registry = FeatureRegistry.open(self.config.registry_path)
        summary: dict[str, Any] = {"autoencoder": None, "gp_features": []}

        if enable_autoencoder:
            windows = self._build_windows(frame, self.config.window)
            ae_cfg = AutoencoderConfig(input_dim=windows.shape[1], **self.config.autoencoder)
            model = MarketAutoencoder(ae_cfg)
            history = model.fit(windows)
            ckpt = model.save(self.config.checkpoint_path)
            final_loss = float(history["reconstruction_loss"][-1]) if history["reconstruction_loss"] else 0.0
            registry.set_autoencoder(str(ckpt), latent_dim=ae_cfg.latent_dim, metrics={"reconstruction_loss": final_loss})
            summary["autoencoder"] = {"checkpoint": str(ckpt), "reconstruction_loss": final_loss}

        if enable_gp:
            gp_cfg = GPFeatureConfig(**self.config.gp)
            discovered = GPFeatureDiscovery(frame, config=gp_cfg).evolve()
            for item in discovered:
                registry.add_gp_feature(item["expression"], metrics={"fitness": item["fitness"]})
            summary["gp_features"] = discovered

        registry.save()
        return summary

"""Automated feature engineering package."""

from phinance.features.autoencoder import AutoencoderConfig, MarketAutoencoder
from phinance.features.extractor import FeatureExtractor
from phinance.features.genetic_features import GPFeatureConfig, GPFeatureDiscovery
from phinance.features.pipeline import FeatureDiscoveryPipeline, FeaturePipelineConfig
from phinance.features.registry import FeatureRegistry

__all__ = [
    "AutoencoderConfig",
    "FeatureExtractor",
    "FeatureDiscoveryPipeline",
    "FeaturePipelineConfig",
    "FeatureRegistry",
    "GPFeatureConfig",
    "GPFeatureDiscovery",
    "MarketAutoencoder",
]

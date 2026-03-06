import numpy as np
import pandas as pd

from phinance.features.extractor import FeatureExtractor
from phinance.features.pipeline import FeatureDiscoveryPipeline, FeaturePipelineConfig


def _market_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    close = 100 + np.cumsum(rng.normal(0, 1, size=n))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": rng.integers(1000, 5000, size=n),
            "iv": np.abs(rng.normal(0.2, 0.03, size=n)),
            "macro_rate": rng.normal(2.0, 0.05, size=n),
        }
    )


def test_feature_discovery_pipeline_end_to_end(tmp_path):
    frame = _market_frame()
    config = FeaturePipelineConfig(
        autoencoder={"latent_dim": 4, "hidden_dim": 12, "epochs": 3, "batch_size": 16, "learning_rate": 0.01},
        gp={"population_size": 8, "generations": 2, "top_k": 2, "random_seed": 3},
        registry_path=str(tmp_path / "feature_registry.json"),
        checkpoint_path=str(tmp_path / "autoencoder_latest.npz"),
        window=8,
    )

    summary = FeatureDiscoveryPipeline(config).run(frame, enable_autoencoder=True, enable_gp=True)
    assert summary["autoencoder"] is not None
    assert len(summary["gp_features"]) > 0

    extractor = FeatureExtractor(config.registry_path, use_autoencoder=True, use_gp_features=True, window=8)
    out = extractor.extract(frame)
    assert out.shape[0] == extractor.output_dim
    assert out.shape[0] >= 6

# Automated Feature Engineering (Phase 7)

Phase 7 adds unsupervised and symbolic feature discovery to Phi-nance.

## Methods

### 1) Autoencoder Latent Features

- Module: `phinance/features/autoencoder.py`
- Trains a lightweight NumPy autoencoder on rolling windows of numeric market data.
- Encoder output (`latent_dim`) becomes a compact learned feature vector.
- Checkpoint format: `.npz` in `phinance/features/checkpoints/`.

### 2) Genetic Programming Features

- Module: `phinance/features/genetic_features.py`
- Uses DEAP symbolic regression primitives to evolve mathematical expressions.
- Fitness is absolute correlation with next-period returns.
- Top expressions are stored in the registry for reuse by agents.

## Registry

- Module: `phinance/features/registry.py`
- Default file: `phinance/features/feature_registry.json`
- Stores:
  - autoencoder checkpoint metadata
  - GP feature expressions + metrics

## Pipeline

- Module: `phinance/features/pipeline.py`
- Config: `configs/features_config.yaml`
- Runs one or both discovery methods and updates registry.

Example:

```python
from phinance.features.pipeline import FeaturePipelineConfig, FeatureDiscoveryPipeline
import pandas as pd

frame = pd.read_parquet("data/bars/SPY_1d.parquet")
cfg = FeaturePipelineConfig.from_yaml("configs/features_config.yaml")
summary = FeatureDiscoveryPipeline(cfg).run(frame, enable_autoencoder=True, enable_gp=True)
print(summary)
```

## Agent Integration (PoC)

- `ExecutionAgent` accepts:
  - `use_auto_features`
  - `use_gp_features`
  - `feature_registry_path`
  - `feature_window`
- `_build_state()` appends discovered features after base and transformer features.

## Data Pipeline Integration

- `DataSourceManager.build_discovered_features(market_data)` computes features if enabled via config.
- Uses cached extractor and existing market window data to control compute/API usage.

## Extending

- Add new methods under `phinance/features/` and register them in `FeatureExtractor`.
- Keep registry entries versioned/annotated with performance metrics.
- Prefer windowed + cached computation to respect free-tier data and CPU constraints.

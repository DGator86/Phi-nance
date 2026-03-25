# Regime layer architecture

There are **three** regime-related packages in this repo. Each has a distinct responsibility; do not mix them.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 · phi.regime  (phi/regime/)                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Training + detection (offline / warmup)                                    │
│                                                                             │
│  • HMMRegimeDetector, ClusteringRegimeDetector, DeepRegimeDetector          │
│  • get_detailed_regime(), get_regime_probabilities()                        │
│  • REGIME_STRATEGY_MAP (BULL_LOW_VOL → playbook list)                       │
│  • phi.force_field  ←── strategy scoring from continuous regime tensor      │
│                                                                             │
│  When to use: train models, backtest regime labels, strategy playbook.      │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ feeds probabilities
┌────────────────────────────▼────────────────────────────────────────────────┐
│  LAYER 2 · regime_engine  (regime_engine/)                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Market Field Theory (MFT) scanner + field math (intraday / live)           │
│                                                                             │
│  • TaxonomyEngine  – KPCOFGS species hierarchy                              │
│  • ProbabilityField – log-space logit propagation                           │
│  • GammaSurface – GEX / dealer positioning surface                         │
│  • Scanner / LiveScanner – real-time feature ➜ logit pipeline               │
│  • IndicatorLibrary, FeatureExtractor                                       │
│                                                                             │
│  When to use: real-time or near-real-time field estimation; QC integration. │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ emits ProjectionPacket
┌────────────────────────────▼────────────────────────────────────────────────┐
│  LAYER 3 · phi.mft  (phi/mft/)  ← formerly src/phinence                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Projection + assignment pipeline (MFT → directional forecast)              │
│                                                                             │
│  • MarketFieldMap (mfm/) – field merger                                     │
│  • engines/ – regime, liquidity, hedge, sentiment, gex_math                 │
│  • Composer  – MFM → ProjectionPacket (direction, drift, vol cones)         │
│  • AssignmentEngine – assigns probability mass to horizons                  │
│  • Store – ParquetBarStore / InMemoryBarStore                               │
│  • Validation – walk-forward + paper trading harness                        │
│                                                                             │
│  Hard boundary: no strategy selection, order routing, or sizing here.       │
│  When to use: produce ProjectionPacket for downstream strategy evaluation.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Canonical import paths

| What you need | Import from |
|---------------|-------------|
| Train/load regime detector | `from phi.regime import HMMRegimeDetector` |
| Discrete playbook (BULL_LOW_VOL → strategies) | `from phi.regime.strategy_mapping import strategies_for_regime` |
| Continuous force-field strategy ranking | `from phi.force_field import run_pipeline` |
| Intraday field scanner / KPCOFGS | `from regime_engine.scanner import Scanner` |
| Gamma surface / GEX | `from regime_engine.gamma_surface import GammaSurface` |
| ProjectionPacket / Composer | `from phi.mft.composer import Composer` |
| Bar store (Parquet / in-memory) | `from phi.mft.store import ParquetBarStore` |
| Assignment engine | `from phi.mft.assignment import AssignmentEngine` |
| QuantConnect export | `from phi.integrations.quantconnect import export_bundle` |

## What **not** to do

- Do **not** import `phi.regime` detectors from inside `regime_engine/` (regime_engine is standalone).
- Do **not** import `phi.mft.*` from `regime_engine/` — the scanner feeds the field map, not the projection pipeline.
- Do **not** add strategy selection logic inside `phi.mft.composer`, `phi.mft.engines`, or `phi.mft.contracts`.

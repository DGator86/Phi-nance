# Data Pipeline Optimisation (Step 5)

This document records profiling, implementation details, and adoption guidance
for the Step 5 data pipeline performance work.

## Baseline profiling

Command used:

```bash
pytest -q tests/performance/test_data_pipeline_profile.py::test_data_pipeline_profile_baseline_vs_optimised
```

Baseline path measured in the profiling test:

1. Build synthetic OHLCV frame.
2. Convert entire frame to numpy.
3. Materialise a copy (simulates fully eager loading).

## Optimisations implemented

### 1. Multi-level optimised cache

- Added `phinance.data.optimised_cache.OptimisedCache`.
- Features:
  - Thread-safe in-memory cache.
  - LRU eviction by estimated byte size.
  - TTL expiration support.
  - Optional disk-cache fallback loading for dataset cache keys.
- `phinance.data.cache.fetch_and_cache(...)` now supports opt-in use via:
  - `use_optimised_cache=True`, or
  - injecting an `optimised_cache` instance.

### 2. Memory-mapped store

- Added `phinance.data.memmap_store.MemmapStore`.
- Persists symbol arrays to `*.dat` + metadata JSON.
- Supports `get_window(start, end)` slicing without eager full-load copies.

### 3. Streaming loader

- Added `phinance.data.streaming_loader.StreamingDataLoader`.
- Streams batches from either:
  - in-memory numpy arrays, or
  - `MemmapStore` symbol memmaps.
- Supports optional shuffling, partial final batch, and prefetch integration.

### 4. Asynchronous prefetcher

- Added `phinance.data.prefetcher.Prefetcher`.
- Background thread fills a bounded queue while compute continues.

### 5. Integration points

- `phinance.live.data_source_manager.DataSourceManager`
  - Optional in-memory `OptimisedCache` from config under
    `data_optimisation.in_memory_cache`.
  - Reuses cache for source payloads and discovered feature payloads.
- `phinance.backtest.engine`
  - Added `simulate_streaming(...)` as opt-in wrapper preserving existing
    default behaviour.
- `phinance.rl.optimised_env_runner`
  - Added `run_steps_prefetched(...)` helper to overlap rollout chunks.

## Config

`configs/data_optimisation_config.yaml`:

```yaml
data_optimisation:
  in_memory_cache:
    enabled: false
    max_size_mb: 1024
    ttl_seconds: 300
  memmap:
    enabled: false
    data_dir: data/memmap
  streaming:
    enabled: false
    batch_size: 256
    prefetch: 2
```

All optimisation paths are opt-in and default to existing behaviour.

## Benchmarks and interpretation

- The performance test captures both baseline and optimised elapsed times in one run.
- It intentionally avoids hard speed assertions because CI/container I/O variance
  can be high.
- Use local repeated runs (`pytest -q -k data_pipeline_profile -s`) to collect
  representative timing for your environment.

## Extending

- For live trading, wire `data_optimisation` section into runtime config loading.
- For RL training, connect `StreamingDataLoader` into dataset episode sampling.
- For feature engineering, cache expensive feature windows with deterministic keys
  (symbol, date range, feature version).

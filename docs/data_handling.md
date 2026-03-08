# Data Handling

## Sources and cache model

Data retrieval is routed through cache-aware helpers so repeated research runs are deterministic and faster.

Typical cache hierarchy:

```text
DATA_CACHE_DIR/
  vendor/
    symbol/
      timeframe/
        *.parquet
        *.metadata.json
```

## Setup workflow

Use `scripts/setup_data_spine.py` to bootstrap local bars and short-volume datasets.

- Can populate from configured providers.
- Can run in sample fallback mode.
- Includes gap checks for data quality.

## Manual refresh

- Re-run setup script with desired symbols/time ranges.
- Clear vendor/symbol folders in `DATA_CACHE_DIR` when forcing cold refresh.
- Keep run artifacts separate from cache for reproducibility.

## Staleness and metadata

Metadata sidecars are used to preserve fetch range and provenance. When changing timeframe/vendor combinations, prefer explicit refetch.

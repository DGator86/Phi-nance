# Meta-Learning for Strategy Discovery

Phase 4 adds a genetic programming (GP) strategy discovery loop under `phinance/meta/`.

## What it does

- Builds a GP primitive set from price/volume and indicator features.
- Evolves expression trees that output a continuous trading signal in `[-1, 1]`.
- Evaluates each candidate with `run_vectorized_backtest`.
- Uses Sharpe-led fitness with drawdown/turnover penalties.
- Stores top strategies in `data/strategy_vault.json`.

## Configuration

Tune `configs/meta_config.yaml`:

- `population_size`
- `generations`
- `cxpb`, `mutpb`
- `min_depth`, `max_depth`
- `top_k`

## Run a discovery campaign

```python
from phinance.meta.search import run_meta_search

result = run_meta_search(ohlcv_df)
print(result["best_strategies"][0])
```

## Agent integration

- `StrategyRDAgent` can load discovered templates from the strategy vault.
- `MetaOrchestrator` can expose discovered strategy options to the meta-policy.

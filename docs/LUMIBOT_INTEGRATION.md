# Lumibot backtesting integration

Phi-nance can run **Lumibot** backtests using the same bar data (Parquet or synthetic) as the native walk-forward backtest. The projection pipeline stays the boundary: Lumibot is a *consumer* of our data and, optionally, of our projection signals.

## Install

```bash
pip install phi-nance[lumibot]
```

## What’s included

| Piece | Purpose |
|-------|--------|
| **`src/phinence/lumibot_bridge/data.py`** | `bar_store_to_pandas_data()` — converts ParquetBarStore/InMemoryBarStore to Lumibot’s `pandas_data` dict (Asset → Data) for `PandasDataBacktesting`. |
| **`src/phinence/lumibot_bridge/strategy.py`** | `create_projection_strategy_class()` — returns a Lumibot Strategy class that runs assign → engines → MFM → composer each day and trades on daily direction (UP/DOWN). |
| **`scripts/run_lumibot_backtest.py`** | CLI: build bar store (synthetic or `--data-root`), build `pandas_data`, run a Lumibot backtest. |

## Commands

```bash
# Buy-and-hold (no projection), synthetic bars, SPY
python -m scripts.run_lumibot_backtest

# Projection strategy: daily direction from our pipeline, trades on UP/DOWN
python -m scripts.run_lumibot_backtest --strategy projection --tickers SPY QQQ --start 2024-01-01 --end 2024-06-30

# Use your bar store
python -m scripts.run_lumibot_backtest --data-root data/bars --strategy projection
```

## Strategies

- **`buy_and_hold`** — Buys 100 shares of the first ticker on the first day. No projection pipeline.
- **`projection`** — Each day runs our pipeline (assign → liquidity/regime/sentiment/hedge → MFM → composer), reads daily direction from `ProjectionPacket`, goes long on UP and sells on DOWN.

## Lumibot 4.x and Tradier

Lumibot 4.x loads its Tradier broker at import time and expects valid Tradier env vars. If you see an error when importing Lumibot (e.g. missing or invalid `TRADIER_TOKEN`), either:

- Set `TRADIER_TOKEN`, `TRADIER_ACCOUNT_NUMBER`, and `TRADIER_IS_PAPER` in `.env` (Lumibot may expect a specific token format), or  
- Use the **native** projection-accuracy backtest instead:  
  `python -m scripts.run_backtest`  
  (no Lumibot; reports AUC and cone coverage only.)

## Boundary (BUILD_SPEC)

Projection logic (engines, MFM, composer) remains inside Phi-nance. Lumibot is used only for:

- Backtest harness (same code path as live in Lumibot’s design).
- Consuming our bar store via `bar_store_to_pandas_data`.
- Optionally consuming our projection output in `ProjectionStrategy` (direction only; no strategy selection, routing, or sizing in the projection layer).

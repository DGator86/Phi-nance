# backtesting.py (kernc) integration

[backtesting.py](https://github.com/kernc/backtesting.py) is a lightweight Python backtesting library — no broker or API keys, just a DataFrame and a Strategy. Phi-nance plugs in our bar store and optional projection pipeline.

## Install

```bash
pip install phi-nance[backtesting]
```

## What’s included

| Piece | Purpose |
|-------|--------|
| **`src/phinence/backtesting_bridge/data.py`** | `bar_store_to_bt_df()` — loads 1m bars for a ticker, resamples to daily (or keeps 1m), returns DataFrame with **Open, High, Low, Close, Volume** and datetime index for `Backtest()`. |
| **`src/phinence/backtesting_bridge/strategy.py`** | `create_projection_strategy()` — returns a `backtesting.Strategy` subclass that runs assign → engines → MFM → composer each bar and goes long on UP / flat on DOWN. |
| **`scripts/run_backtesting_py.py`** | CLI: build bar store (synthetic or `--data-root`), build OHLCV DataFrame, run `Backtest(df, Strategy, ...).run()`. |

## Commands

```bash
# SMA crossover (built-in example), synthetic SPY
python -m scripts.run_backtesting_py

# Projection strategy (our pipeline drives signals)
python -m scripts.run_backtesting_py --strategy projection --ticker SPY --start 2024-01-01 --end 2024-06-30

# Your bar store, no plot window
python -m scripts.run_backtesting_py --data-root data/bars --strategy projection --no-plot
```

## Strategies

- **`sma_cross`** — 10/20 SMA crossover (from backtesting.py examples). No projection pipeline.
- **`projection`** — Each day runs our pipeline (assign → liquidity/regime/sentiment/hedge → MFM → composer), reads daily direction from `ProjectionPacket`, buys on UP and closes position on DOWN.

## Why backtesting.py vs Lumibot

| | backtesting.py | Lumibot |
|---|----------------|--------|
| **Install** | `pip install backtesting` | `pip install lumibot` (heavier) |
| **Broker/keys** | None | Can require Tradier/Polygon for import |
| **Data** | DataFrame (OHLCV) | DataFrame or Yahoo/Polygon/etc. |
| **API** | `Backtest(data, Strategy).run()` | `Strategy.run_backtest(DataSource, start, end, ...)` |
| **License** | AGPL-3.0 | Check Lumibot repo |

Use **backtesting.py** when you want fast, key-free backtests with our data and/or projection signals. Use **Lumibot** when you want the same strategy code path for backtest and live trading.

## Boundary (BUILD_SPEC)

Projection logic stays inside Phi-nance; backtesting.py is only the execution harness. Strategy uses direction from `ProjectionPacket` only (no strategy selection, routing, or sizing in the projection layer).

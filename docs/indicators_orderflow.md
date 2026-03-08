# Order Flow & Liquidity Indicators

This document describes the order-flow indicator layer added to `phi.indicators`.

## Overview

The platform now supports order-flow-aware indicators with a provider abstraction:

- `OrderFlowProvider` defines `get_order_flow(ohlcv) -> DataFrame`
- `OHLCVOrderFlowProvider` provides a heuristic fallback derived from OHLCV bars
- A global provider can be set via:
  - `phi.indicators.orderflow.set_order_flow_provider(provider)`
  - `phi.indicators.orderflow.get_order_flow_provider()`

## Data contract

Order flow providers return a DataFrame aligned to `ohlcv.index` with columns:

- `buy_volume`
- `sell_volume`
- `cumulative_delta`
- `tick_count`
- `bid`
- `ask`
- `spread`
- `depth`

## Heuristic provider (OHLCV)

`OHLCVOrderFlowProvider` uses candle direction:

- `close > open` -> all bar volume counted as buy volume
- `close < open` -> all bar volume counted as sell volume
- `close == open` -> volume split 50/50

This is a practical bootstrapping method for backtests when tick data is unavailable.

## Indicators

The following new registry indicators are available under type `orderflow`:

1. **Order Flow VWAP** (`orderflow_vwap`)
   - Signal: `(close - vwap) / ATR`, clipped and normalized to `[-1, 1]`
   - Params: `atr_period`, `clip_value`

2. **Volume Profile** (`volume_profile`)
   - Rolling bins over close prices and volume to estimate Point of Control (POC)
   - Signal: `+1` near POC, `-1` otherwise
   - Params: `window`, `bins`, `near_poc_threshold`

3. **Cumulative Delta** (`cumulative_delta`)
   - Delta = `buy_volume - sell_volume`
   - Signal: rolling cumulative delta normalized by rolling total volume
   - Params: `window`, `clip_value`

4. **Liquidity Metrics** (`liquidity`)
   - Uses spread (or spread proxy) + Amihud illiquidity + relative volume
   - Signal normalized to `[-1, 1]` via `tanh`
   - Params: `window`, `amihud_scale`

## Streamlit usage

The workbench indicator selector includes these under **Order Flow & Liquidity**:

- Orderflow VWAP
- Volume Profile
- Cumulative Delta
- Liquidity Metrics

## Adding a real tick provider

To integrate true order-flow data:

1. Implement a new class inheriting `OrderFlowProvider`.
2. Fill the standard schema columns from your tick/L2 source.
3. Set it at runtime:

```python
from phi.indicators.orderflow import set_order_flow_provider
set_order_flow_provider(MyTickOrderFlowProvider(...))
```

No indicator code changes are required if your provider respects the schema.

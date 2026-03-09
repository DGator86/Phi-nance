# Multi-Asset Portfolio Backtesting

This document describes the portfolio backtesting flow for running multi-symbol simulations.

## Overview

The engine supports:

- Multiple symbols via `RunConfig.symbols`.
- Portfolio-level accounting (cash + open positions).
- Allocation strategies:
  - `equal_weight`
  - `fixed_weight`
  - `signal_weighted`
  - `risk_parity` (inverse volatility)
- Rebalancing:
  - Time-based: `D`, `W`, `M`, `Q`, or integer bars.
  - Threshold-based drift trigger via `rebalance_threshold`.

## Core Components

- `phi/backtest/portfolio.py`: `Portfolio` and `Order` models.
- `phi/backtest/allocation.py`: allocation strategy interface and implementations.
- `phi/backtest/direct.py`: `run_portfolio_backtest` engine.

## Streamlit Configuration

In the backtest form:

1. Enter symbols as a comma-separated list.
2. Choose allocation strategy.
3. For `fixed_weight`, provide one weight per symbol (sum should be `1.0`).
4. Select rebalance frequency.
5. Optionally enable threshold rebalancing.

## Results

The UI presents:

- Portfolio metrics (return, CAGR, Sharpe, max drawdown).
- Portfolio equity curve.
- Transaction log.
- Per-symbol contribution summary.

## Extending Allocation

Create a class inheriting `AllocationStrategy` and implement:

```python
allocate(capital, signals, prices, **kwargs) -> dict[str, float]
```

Then register it in `get_allocation_strategy`.

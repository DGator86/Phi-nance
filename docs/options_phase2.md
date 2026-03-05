# Options Module Phase 2

Phase 2 adds:

- IV surface construction (`phi/options/iv_surface.py`).
- Strategy primitives (`phi/options/strategies/*`).
- Options portfolio backtesting engine (`phi/backtest/options_engine.py`).
- Workbench Options UI stub (`app_streamlit/live_workbench.py`).

## IV Surface

Build from a chain DataFrame with `strike`, `expiry` (years), and `iv`:

```python
surface = IVSurface(chain_df)
iv = surface.get_iv(strike=102.5, expiry=0.18)
```

## Strategies

Use concrete strategies from `phi.options.strategies`:

- `SingleLeg`
- `VerticalSpread`
- `Straddle`
- `Strangle`
- `IronCondor`

Each strategy provides:

- `legs()`
- `validate()`
- `net_premium(S, r, iv_surface)`
- `greeks(S, r, iv_surface)`

## Backtesting

`OptionsBacktestEngine` consumes `RunConfig` with `options_strategies` and optional `iv_chain_data`.

If no IV chain is supplied, engine falls back to constant IV from `default_iv` (default `0.20`).

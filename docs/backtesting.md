# Backtesting

## Backtest paths

- CLI: `run_backtest.py`
- Script module: `scripts/run_backtest.py`
- Streamlit-integrated execution from `app_streamlit`
- Options-specific flows under `phi/options/backtest.py`

## Output artifacts

Backtests typically emit:

- run configuration snapshot
- summary metrics
- trade history/details
- logs and optional debug artifacts

## Extending engines

1. Add/modify strategy logic in `strategies/`.
2. Ensure data contracts are satisfied by adapters.
3. Add tests for deterministic behavior.
4. Document new controls or outputs.

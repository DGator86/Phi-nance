# Options Backtesting

Phi-nance includes options-oriented modules under `phi/options/`.

## Scope

- Option pricing/greeks helpers
- Strategy building blocks
- Backtest integration path for options mode

## Current constraints

- Designed as a practical research framework rather than full exchange-grade simulation.
- Some workflows use simplified assumptions where full chain depth is unavailable.
- Optional vendor enrichments (for example `MARKETDATAAPP_API_TOKEN`) can improve realism.

## Getting started

- Enable options trading mode in Streamlit workbench.
- Configure data keys in `.env` as needed.
- Run backtests and inspect run artifacts under `${RUNS_DIR}`.

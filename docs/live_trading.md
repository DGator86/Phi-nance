# Live Trading Integration

## Overview

The `phi.live` package provides broker-agnostic live/paper trading execution:

- `phi/live/broker.py`: broker abstraction + Alpaca adapter
- `phi/live/portfolio.py`: real-time portfolio/equity tracking
- `phi/live/strategy.py`: signal/allocation-driven order generation
- `phi/live/risk.py`: position and loss risk limits
- `phi/live/engine.py`: main loop and bar processing
- `phi/live/loader.py`: latest optimized config loading

## Setup

1. Copy `.env.example` to `.env`.
2. Fill Alpaca keys:
   - `BROKER_API_KEY`
   - `BROKER_SECRET_KEY`
   - `BROKER_BASE_URL` (`https://paper-api.alpaca.markets` for paper)
3. Configure symbols and cadence:
   - `LIVE_SYMBOLS=SPY,QQQ`
   - `LIVE_UPDATE_INTERVAL=60`

## Deploy best configuration

- Auto-train and deploy in one command:

```bash
python scripts/auto_train.py --tickers SPY --deploy-live
```

- Or deploy manually:

```bash
python scripts/deploy_live_config.py --best-dir runs/best_params
```

## Start trader

```bash
python scripts/start_live_trader.py --config live_config.json
```

## Streamlit monitoring

Open the Streamlit app and switch to **Live Trading** page in the sidebar.
The page shows account, positions, open orders, actions, and equity curve.

## Safety notes

- Start with `LIVE_MODE=paper`.
- Configure conservative risk limits before enabling live capital.
- Monitor drawdowns and stop engine on unexpected behavior.

# Limit Order Book Simulation

This module adds event-driven LOB simulation for tick-level backtesting.

## Data formats

Standard event schema:

- `timestamp`
- `event_type` (`add`, `cancel`, `trade`, `quote`)
- `price`
- `volume`
- `side` (`buy`/`sell`)
- optional `order_id`

Loaders are provided in `phi/lob/data.py` for:

- LOBSTER CSV (`load_lobster_csv`)
- Dukascopy ticks (`load_dukascopy_ticks`)
- Custom CSV mappings (`load_custom_csv`)

## Engine

`LobSimEngine` replays events, updates `OrderBook`, calls a `LobStrategy`, executes
returned market/limit/cancel orders, and tracks performance with `phi.backtest.portfolio.Portfolio`.

Outputs include:

- fills/trade log
- equity curve
- summary metrics (PnL, return, sharpe, trades)

## Writing a strategy

Subclass `LobStrategy` and implement:

- `on_event(event, book, portfolio) -> list[SimOrder]`

Optional hooks:

- `on_start(book, portfolio)`
- `on_finish(book, portfolio)`

Reference strategies:

- `MarketMakingStrategy`
- `ImbalanceStrategy`

## Streamlit UI

Use `app_streamlit/main.py`, select **LOB Simulation** page.

Features:

- Source selection: Synthetic / LOBSTER / Dukascopy / Custom CSV
- Strategy selection and parameter controls
- Results: metrics, equity curve, depth snapshot, fills log

## CLI

Run:

```bash
python scripts/run_lob_sim.py --data synthetic --strategy market_making --max-events 500
```

Useful options:

- `--data synthetic|lobster|dukascopy|custom`
- `--path <csv file>`
- `--strategy market_making|imbalance`
- `--params '{"size": 2.0, "threshold": 0.3}'`
- `--output result.json`

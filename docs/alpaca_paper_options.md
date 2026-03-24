# Alpaca paper trading (options) + PDT + MTF

## Security first

- **Never commit API keys.** Use `.env` (gitignored) or your OS secret store.
- If keys were pasted into chat, email, or a ticket, **rotate them in the Alpaca dashboard** and update `.env` only on your machine.

## Environment variables

The health script and `phinance.live.alpaca.AlpacaBroker` accept either naming style:

| Variable | Example |
|----------|---------|
| `ALPACA_API_KEY` | Paper key starting with `PK…` |
| `ALPACA_SECRET_KEY` | Secret (shown once in dashboard) |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` |

Aliases (from `phi.config` / `.env.example`):

- `BROKER_API_KEY`, `BROKER_SECRET_KEY`, `BROKER_BASE_URL`
- `BROKER_ACCOUNT_ID` (optional)

**Two paper accounts (e.g. PDT vs non-PDT):** you can keep both in local `.env` using profile-prefixed vars and switch with `BROKER_PROFILE`.

```bash
BROKER_PROFILE=PHINANCE_PDT
BROKER_PHINANCE_PDT_ACCOUNT_ID=PAxxxxxxxxxx
BROKER_PHINANCE_PDT_API_KEY=PK...
BROKER_PHINANCE_PDT_SECRET_KEY=...
BROKER_PHINANCE_PDT_BASE_URL=https://paper-api.alpaca.markets/v2

BROKER_PAPER_2_ACCOUNT_ID=PAyyyyyyyyyy
BROKER_PAPER_2_API_KEY=PK...
BROKER_PAPER_2_SECRET_KEY=...
BROKER_PAPER_2_BASE_URL=https://paper-api.alpaca.markets/v2
```

You can still use a single default `BROKER_*` set. Never commit real keys.

## $5k account and PDT

- **Pattern Day Trader (PDT)** (US, margin): if equity **&lt; $25,000**, you are generally limited to **3 day trades** in **5 rolling business days** when using a margin account. **Options** trades count toward day trades when you open and close the same option on the same day.
- **Cash account** rules differ (settled funds); confirm your Alpaca account type in the dashboard.
- Alpaca’s paper account may show `pattern_day_trader` and related fields—use the health script to inspect.

This project does **not** auto-enforce PDT for you; size and hold time are your responsibility.

## Options only

- Alpaca supports listed options in paper/live subject to their contracts and market data entitlements.
- The bundled `phinance.live.alpaca.AlpacaBroker` today focuses on **equity** order helpers. **Option legs** (OCC symbols, multi-leg) require Alpaca’s options order API; add a thin wrapper when you are ready—start with **manual** small tests in paper after the health check passes.

## Confirm MTF (multi-timeframe regimes)

Phi-nance MTF for **daily** underlying history is implemented in `phi/regime/mtf_matrix.py` and used by the **Trading desk → Build signal** card.

**Facts:**

1. OHLCV is loaded **Unusual Whales first**, then **yfinance** (`phi.data.fetch_ohlcv_uw_then_yf`).
2. With **daily** bars, pandas offsets **finer than 1D** (1m … 4H) are **skipped**; you typically see **1D, 1W, 1ME** (plus similar) when enough history exists after resampling.
3. **Confluence** is the equal-weight mean of semantic regimes mapped to +1 / 0 / −1 (`TREND_UP` / `RANGE` / `TREND_DN`).

Run from repo root (venv activated):

```bash
python scripts/alpaca_paper_options_health.py --confirm-mtf --symbol SPY
```

That hits **Alpaca only for the account snapshot**; MTF uses the same **UW → yfinance** daily series as the app (not Alpaca intraday bars).

To align MTF with **Alpaca intraday** bars later, add a fetch path that builds a minute/hour DataFrame and pass it into `build_regime_matrix`.

## Health check (no orders)

```bash
python scripts/alpaca_paper_options_health.py
```

Expected: equity, cash, buying power, `pattern_day_trader`, etc. **No orders are placed.**

## Dependencies

`alpaca-py` is listed in `requirements.txt`. If needed:

```bash
pip install alpaca-py
```

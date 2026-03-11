# PostgreSQL Options Vendor

Phi-nance includes a `postgres` data vendor for loading intraday options rows from a PostgreSQL database.

## Environment variables

Set these in `.env`:

- `POSTGRES_HOST` (default `localhost`)
- `POSTGRES_PORT` (default `5432`)
- `POSTGRES_DB` (default `optionsdata`)
- `POSTGRES_USER` (default `postgres`)
- `POSTGRES_PASSWORD`

## Expected schema

- One table per symbol (for example `spy`, `fb`).
- Table is resolved from `symbol.lower()` with non-alphanumeric characters stripped.
- Timestamp column defaults to `quote_time` and defaults to milliseconds since epoch.

If your schema differs, pass overrides in fetch kwargs:

```python
fetch_and_cache(
    vendor="postgres",
    symbol="SPY",
    start="2022-01-10",
    end="2022-01-15",
    timestamp_column="quote_time",
    timestamp_unit="ms",  # s | ms | ns
)
```

## Backtesting adapter

Use `phi.options.data_adapter.adapt_for_backtesting` (or `fetch_options_data`) to normalize raw options fields into a strategy-friendly frame with datetime index and expected columns.

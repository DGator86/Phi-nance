# Options Module – Experimental

This module is under active development. Pricing models and Greeks are not yet fully productionized.
Use at your own risk. See `phi/options/TODO.md` for progress.

## PostgreSQL options vendor quickstart

1. Install runtime/dev dependencies:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`
2. Configure database credentials in `.env`:
   - `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
3. Smoke test the vendor:
   - Call `phi.data.cache.fetch_and_cache(vendor="postgres", symbol="SPY", start="2023-01-10", end="2023-01-15")`.
4. Normalize rows for research:
   - Run `phi.options.data_adapter.adapt_for_backtesting(df)` and confirm a `DatetimeIndex` plus option fields (delta/gamma/open_interest).
5. Validate hardening tests:
   - `pytest tests/phi/data/test_vendor_postgres.py tests/phi/options/test_data_adapter.py -v`
6. Start strategy scaffolding:
   - See `strategies/my_options_strategy.py` for a minimal end-to-end signal generator using `fetch_options_data()`.

### Troubleshooting

- Table naming: the postgres vendor resolves table names from `symbol.lower()` (for `SPY`, table should be `spy`).
- Timestamp conversion: default assumes milliseconds; pass `timestamp_unit="s"` if the DB stores seconds.
- Missing columns: `adapt_for_backtesting` injects defaults, but you can extend field mapping in `phi/options/data_adapter.py`.

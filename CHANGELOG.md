# Changelog

All notable changes to Phi-nance are documented here.

## [0.1.0] - 2026-02-25

### Added
- Comprehensive CI pipeline with GitHub Actions (lint + tests)
- Unit tests for `phi/data/cache.py`, `phi/blending/blender.py`, `phi/run_config.py`, `phi/options/backtest.py`
- Error handling with retry logic (exponential backoff, max 3 retries) for all API calls (Alpha Vantage, yFinance, Binance)
- OHLCV data sanity validation after fetch (negative price check, chronological order check)
- Cache staleness detection (`fetched_at` timestamp in metadata; warn if data older than 1 day for intraday, 7 days for daily)
- `manage_cache.py` CLI utility for cache cleanup (`--clean --days N`) and listing (`--list`)
- `compute_greeks(delta, gamma, theta, vega)` helper in `phi/options/backtest.py`
- Parallel indicator tuning in `run_phiai_optimization()` via `ThreadPoolExecutor`
- Expanded `PhiAI.explain()` with richer configuration + change summary output
- Vectorized `regime_weighted` blend method in `phi/blending/blender.py`
- Input validation in Streamlit workbench (`st.error()` for invalid date ranges, empty symbols)
- Resource warnings in Streamlit for date ranges > 5 years or > 4 indicators
- Expanded README with Usage examples, RunConfig schema, and Customization guide
- `DEPLOYMENT.md` consolidating VPS, Oracle Cloud, and local setup guides
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- Logging throughout `phi/data/cache.py`

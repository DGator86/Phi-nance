# Live Trading Integration (Free Data Sources)

This phase adds a **rate-limit resilient** live trading stack for paper trading.

## Components

- `phinance.live.rate_limiter.RateLimiter`: token-bucket throttling per source.
- `phinance.live.cache.PersistentCache`: SQLite cache with TTL.
- `phinance.live.data_source_manager.DataSourceManager`: source priority, fallback, usage telemetry.
- `phinance.live.engine.LiveEngine`: runtime orchestration with fail-safe halting.
- `phinance.live.order_manager.OrderManager`: kill-switch and risk check gate.
- `app_streamlit/live_dashboard.py`: account + data source health panel.

## Source strategy

- Use configured `data_priorities` in `configs/live_config.yaml`.
- Check cache before API call.
- Apply per-source token bucket.
- Skip overloaded sources and fallback.
- Expose `usage_snapshot()` for dashboard/alerts.

## Safety

- Manual kill switch available through `OrderManager.set_kill_switch(True)`.
- Engine halts when required market data is unavailable.

## Environment variables

Keep API keys in `.env`:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `THETADATA_API_KEY`
- `FRED_API_KEY`
- `FMP_API_KEY`

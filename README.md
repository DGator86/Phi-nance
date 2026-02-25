# Phi-nance — Live Backtest Workbench

Quantitative trading research platform with regime-aware MFT (Market Field Theory) and a premium Live Backtest Workbench.

---

## Quick Start

```bash
# Run the Live Backtest Workbench (recommended)
python -m streamlit run app_streamlit/live_workbench.py

# Or the legacy MFT Dashboard
python -m streamlit run dashboard.py

# CLI backtest
python run_backtest.py --strategy rsi --start 2020-01-01 --end 2024-12-31 --budget 100000
```

---

## Live Backtest Workbench

1. **Dataset Builder** — Fetch & cache OHLCV (Alpha Vantage, yfinance for daily)
2. **Indicator Selection** — RSI, MACD, Bollinger, Dual SMA, Mean Reversion, Breakout, Buy & Hold
3. **Blending Panel** — Weighted Sum, Regime-Weighted, Voting, PhiAI Chooses
4. **PhiAI Panel** — Full auto mode for indicator/param/blend optimization
5. **Backtest Controls** — Equities/Options, position sizing, exit rules
6. **Run & Results** — Live progress, metrics, Run History, Cache Manager

**Initial Capital** is required and validated. All runs are stored under `runs/{run_id}/` with config, results, and trades.

---

## What Was Added (Master Prompt)

| Area | Original | Added |
|------|----------|-------|
| **Data** | Alpha Vantage, dashboard cache | `phi.data` — `/data_cache/{vendor}/{symbol}/{timeframe}/` parquet + metadata |
| **Runs** | None | `phi.run_config` — RunConfig schema, RunHistory at `/runs/` |
| **UI** | 6-tab dashboard | `app_streamlit/live_workbench.py` — step-by-step Workbench |
| **Blending** | MFT blender in dashboard | `phi.blending` — Weighted Sum, Voting, Regime-Weighted |
| **PhiAI** | None | `phi.phiai` — auto_tune_params, PhiAI orchestrator |
| **Options** | None | `phi.options` — stub for future options backtest |
| **Theme** | Default | Dark only, purple (#a855f7) + orange (#f97316) |

---

## Module Map

| Module | Location |
|--------|----------|
| Data cache | `phi/data/cache.py` |
| RunConfig + RunHistory | `phi/run_config.py` |
| Blending | `phi/blending/blender.py` |
| PhiAI | `phi/phiai/auto_tune.py` |
| Options (stub) | `phi/options/` |
| Live Workbench | `app_streamlit/live_workbench.py` |
| Architecture | `Architecture.md` |


## External Ecosystem Notes

A curated landscape of external options/data projects (with recommended integration order for Phi-nance) is maintained in:

- `docs/external-options-data-landscape.md`

---

## Dependencies

See `requirements.txt`. Key: lumibot, streamlit, pandas, yfinance, scikit-learn, lightgbm.

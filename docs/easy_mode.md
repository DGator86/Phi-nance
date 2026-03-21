# Phi-nance easy mode

The default app (`app_streamlit/main.py` → `live_workbench.py`) is a **three-screen** experience:

1. **Overview** — Optional nightly learning JSON, recent saved runs, universe regime snapshot.
2. **Automatic backtest** — One button: full-catalog regime-weighted run (same as expert stack), three signal-threshold presets, best Sharpe highlighted. **Visual replay** for the winning preset: price + **equity on a second Y-axis**, **regime shading** (mapped TREND_DN / RANGE / TREND_UP), **buy/sell markers** with hover text (`composite>threshold`, `eod_liquidation`, etc.), plus **context cards** (model regime, 5-bar and 20-bar thrust). After each run: **SHORT / MEDIUM / LONG** k-means tiers (different feature windows), **playback** slider (truncate to an end index), **compare** the best preset side-by-side with another, **robustness** (bootstrap Sharpe from shuffled bar returns), and **RSI × threshold heatmap** (grid over a small probe stack). Expandable **trade log** and **last-bar indicator snapshot** (top contributors by absolute signal). Uses a kwargs shim so older `run_direct_backtest` signatures still run when regime args are missing.
3. **Ticker spotlight** — Pick a universe ticker; price, regime lane, RSI & MACD signals, plus a **countdown to the next US equity daily cash close** (4pm ET, weekdays).

## Optional extras

- **Dash wallboard** (`app_dash/app.py`): SPY daily chart, 60s refresh. Install: `pip install -r requirements-dash.txt`.
- **Notebook path** (`notebooks/10_easy_strategy_lab.ipynb`): same data + multi-window regime helper without Streamlit.

## Environment

| Variable | Purpose |
|----------|---------|
| `PHINANCE_UNIVERSE` | Comma-separated tickers (default `SPY,QQQ,IWM,DIA,GLD`) |
| `PHINANCE_EASY_LOOKBACK` | Calendar days of history (default `420`) |
| `PHINANCE_EASY_PRIMARY` | Symbol for automatic backtest (default `SPY`) |
| `UNUSUAL_WHALES_API_KEY` | Preferred OHLC source; falls back to Yahoo |

## Overnight learning file

Cron (or any job) can write:

`{DATA_CACHE_DIR}/phi_nance_learning_summary.json`

Example:

```json
{
  "updated_at": "2026-03-21T06:00:00Z",
  "cycle": "nightly",
  "notes": "PhiAI sweep completed; promoted balanced threshold.",
  "metrics": { "sharpe": 1.05, "total_return": 0.12 },
  "per_ticker": {
    "SPY": { "regime": "BULL_NORMAL_VOL", "note": "Add size on dips" }
  }
}
```

## Expert workbench

Full options, regime training UI, LOB, live trading:

```bash
streamlit run app_streamlit/expert_workbench.py
```

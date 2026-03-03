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

1. **Dataset Builder** — Fetch & cache OHLCV (Alpha Vantage, yfinance, Binance Public)
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
| **Options** | None | `phi.options` — delta backtest + optional MarketDataApp snapshot |
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


## Optional API keys

- `MARKETDATAAPP_API_TOKEN`: enables real options-chain snapshots in options mode to tune delta assumptions when available.

---

## Usage

### Data Fetching

```python
from phi.data.cache import fetch_and_cache

# Fetch and cache daily SPY data from yfinance
df = fetch_and_cache(
    vendor="yfinance",
    symbol="SPY",
    timeframe="1D",
    start="2022-01-01",
    end="2024-01-01",
)
print(df.tail())
```

### Building a Blended Strategy

```python
import pandas as pd
from phi.blending.blender import blend_signals

# signals is a DataFrame where each column is an indicator signal series
signals = pd.DataFrame({
    "RSI": rsi_signal,    # values in [-1, 1]
    "MACD": macd_signal,
})
composite = blend_signals(signals, method="weighted_sum", weights={"RSI": 0.6, "MACD": 0.4})
```

### Running PhiAI Auto-Tuning

```python
from phi.phiai.auto_tune import run_phiai_optimization

indicators = {
    "RSI": {"enabled": True, "auto_tune": True, "params": {}},
    "MACD": {"enabled": True, "auto_tune": True, "params": {}},
}
optimized, explanation = run_phiai_optimization(ohlcv, indicators, max_iter_per_indicator=20)
print(explanation)
```

---

## RunConfig Schema

All backtest runs are described by a `RunConfig` object and saved to `runs/{run_id}/`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | str | `""` | Identifier for the dataset used |
| `symbols` | list[str] | `["SPY"]` | List of ticker symbols |
| `start_date` | str | `""` | ISO date string `YYYY-MM-DD` |
| `end_date` | str | `""` | ISO date string `YYYY-MM-DD` |
| `timeframe` | str | `"1D"` | Bar timeframe: `1D`, `1H`, `15m`, etc. |
| `vendor` | str | `"alphavantage"` | Data vendor key |
| `initial_capital` | float | `100000.0` | Starting portfolio capital |
| `trading_mode` | str | `"equities"` | `"equities"` or `"options"` |
| `indicators` | dict | `{}` | Indicator configs `{name: {enabled, params}}` |
| `blend_method` | str | `"weighted_sum"` | Signal blend method |
| `blend_weights` | dict | `{}` | Per-indicator blend weights |
| `phiai_enabled` | bool | `False` | Enable PhiAI auto-tuning |
| `evaluation_metric` | str | `"roi"` | Optimization metric |

**Saved run structure:**
```
runs/{run_id}/
  config.json    ← RunConfig serialized
  results.json   ← Metrics: total_return, cagr, sharpe, max_drawdown
  trades.csv     ← Trade log (if available)
```

---

## Customization

### Adding a New Indicator

1. Create `strategies/my_indicator.py` with a Lumibot `Strategy` subclass.
2. Register it in `app_streamlit/live_workbench.py` in `INDICATOR_CATALOG`.
3. Add a parameter grid entry in `phi/phiai/auto_tune.py` (`param_grids`).
4. Add a `REGIME_INDICATOR_BOOST` entry in `phi/blending/blender.py`.

### Adding a New Blender

1. Add a new method name to `BLEND_METHODS` in `phi/blending/blender.py`.
2. Implement the logic as an `if method == "my_method":` branch in `blend_signals()`.
3. Return a `pd.Series` with the same index as `signals`.

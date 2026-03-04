# Phi-nance — Architecture

## What this repo originally had

| Module | Description |
|---|---|
| `dashboard.py` | Streamlit multi-tab dashboard (7 tabs: ML Status, Fetch Data, MFT Blender, Phi-Bot, Plutus Bot, Backtests, System Status) |
| `strategies/` | 15+ Lumibot-backed strategies (BuyAndHold, RSI, MACD, Bollinger, DualSMA, MeanReversion, MomentumRotation, ChannelBreakout, Wyckoff, LiquidityPools, WyckoffML, PhiBot) |
| `regime_engine/` | Full MFT (Market Field Theory) engine: features, taxonomy, probability field, indicator library, options engine, parameter tuner, gamma surface, L2 client, data fetcher |
| `run_backtest.py` | CLI backtest runner (Lumibot / Alpha Vantage) |
| `train_ml_classifier.py` | ML regime classifier training |
| `train_rl_agent.py` | RL agent training |
| `data/` | Simple data cache |
| `models/` | Persisted ML/RL model files |

---

## What was added — the Live Backtest Workbench

### New: `phi/` — Core Engine Package

```
phi/
├── __init__.py
├── data/
│   ├── cache.py          # Dataset cache: parquet + metadata JSON
│   └── fetchers.py       # yfinance (primary) + Alpha Vantage fetcher
├── indicators/
│   └── registry.py       # 14 indicators with compute functions + tune ranges
├── blending/
│   └── blender.py        # weighted_sum | voting | regime_weighted | phiai
├── backtest/
│   ├── run_config.py     # RunConfig dataclass (full reproducibility)
│   ├── run_history.py    # Save/load/compare runs under /runs/{run_id}/
│   └── engine.py         # Vectorized bar-by-bar backtest engine
├── options/
│   └── simulator.py      # Black-Scholes options backtest simulation
└── phiai/
    └── tuner.py          # Random-search auto-tuner + indicator selector
```

### New: `app_streamlit/` — Workbench UI

```
app_streamlit/
├── __init__.py
├── styles.py             # Dark purple/orange CSS theme
└── live_workbench.py     # Full 5-step workbench + results + history + cache manager
```

### New: Storage Directories

```
data_cache/
  {vendor}/{symbol}/{timeframe}/{start}_{end}.parquet
  {vendor}/{symbol}/{timeframe}/{start}_{end}_metadata.json

runs/
  {run_id}/
    config.json           # Full RunConfig JSON
    results.json          # Metrics + equity curve
    trades.csv            # Per-trade log
```

---

## Module Responsibilities

### `phi/data/cache.py`
- `is_cached(vendor, symbol, timeframe, start, end)` → bool
- `load_dataset(...)` → DataFrame or None
- `save_dataset(df, ...)` → Path (saves parquet + metadata.json)
- `list_cached_datasets()` → list of metadata dicts
- `clear_all_cache()` → int (count removed)

### `phi/data/fetchers.py`
- `fetch(symbol, start, end, timeframe, vendor)` → standardized OHLCV DataFrame
- Supports: `1D, 4H, 1H, 15m, 5m, 1m` timeframes
- Vendors: `yfinance` (primary), `alpha_vantage` (secondary)

### `phi/indicators/registry.py`
| Indicator | Type | Description |
|---|---|---|
| `rsi` | A | RSI oscillator → buy when oversold |
| `macd` | B | MACD histogram momentum |
| `bollinger` | A | Bollinger band position |
| `stochastic` | A | Stochastic %K/%D |
| `dual_sma` | B | SMA golden/death cross |
| `ema_crossover` | B | EMA fast/slow cross |
| `momentum` | B | Raw price momentum |
| `roc` | B | Rate of Change |
| `atr_ratio` | B | ATR vs its MA (volatility) |
| `vwap_dev` | D | VWAP deviation (mean-reversion) |
| `cmf` | A | Chaikin Money Flow |
| `adx` | B | ADX × DI direction/strength |
| `wyckoff` | C | Volume-price divergence |
| `range_pos` | A | Rolling range position |
| `phi_mft` | MFT | Full MFT regime composite |

All indicators output normalized signals in **[-1, +1]** where +1 = strong buy, -1 = strong sell.

### `phi/blending/blender.py`
- `weighted_sum` — linear combination of signals
- `voting` — weighted majority vote
- `regime_weighted` — per-regime weight matrix × MFT probabilities
- `phiai` — weight by individual indicator backtest score

### `phi/backtest/engine.py`
Bar-by-bar simulation (vectorized with NumPy):
1. Signal → raw position series (threshold-gated)
2. Entry on position change
3. Exit on: stop loss / take profit / trailing stop / time exit / signal reversal
4. Equity curve tracking
5. Metrics: return, CAGR, Sharpe, Sortino, Calmar, max DD, Profit Factor, Win Rate, Direction Accuracy

### `phi/backtest/run_config.py`
`RunConfig` dataclass captures:
- Dataset (id, symbol, timeframe, start/end, vendor)
- Initial capital
- Trading mode (equities / options)
- Indicators (list of `IndicatorConfig` with params, auto_tune flag)
- Blend mode + weights + regime weights
- PhiAI toggles + constraints
- Exit rules (SL, TP, trailing stop, time exit, signal exit)
- Position sizing (method, pct, allow short)
- Options config (structure, DTE, profit/stop exit)
- Evaluation metric
- Description / tags

Serialized to `config.json` for every run.

### `phi/phiai/tuner.py`
- `PhiAITuner.select_indicators()` — scores each indicator individually, selects top-N
- `PhiAITuner.tune_indicator()` — random search over parameter space
- `phiai_full_auto()` — convenience wrapper for the workbench

### `phi/options/simulator.py`
Simplified options P&L simulation:
- Uses rolling realized volatility as IV proxy
- Black-Scholes pricing for calls/puts/spreads
- Entry on signal threshold, exit on profit/stop/expiry/signal reversal

---

## Data Flow

```
User Input (symbol, dates, timeframe, capital)
        ↓
   Data Fetcher (yfinance / Alpha Vantage)
        ↓
   Dataset Cache (/data_cache/...)
        ↓
   Indicator Registry (compute signals [-1, +1])
        ↓                    ↑ (optional)
   PhiAI Tuner ────────────┘ (auto-select + tune params)
        ↓
   Blender (weighted_sum / voting / regime_weighted / phiai)
        ↓
   Blended Signal Series
        ↓
   Backtest Engine (equities) OR Options Simulator
        ↓
   BacktestResult (equity_curve, trades, metrics)
        ↓
   Run History (/runs/{run_id}/)
        ↓
   Workbench UI (results tabs, comparison, export)
```

---

## Launch Commands

```bash
# New workbench (recommended)
streamlit run app_streamlit/live_workbench.py

# Legacy dashboard (preserved)
streamlit run dashboard.py

# CLI backtest (existing)
python run_backtest.py --strategy rsi --budget 50000
```

---

## Theme

Dark mode enforced via:
- `.streamlit/config.toml` — `base = "dark"`, `primaryColor = "#a855f7"` (purple)
- `app_streamlit/styles.py` — comprehensive CSS injection
- Color palette: Purple `#a855f7` / Orange `#f97316` / Charcoal `#0a0a12`

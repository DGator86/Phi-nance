# Phi-nance Architecture

## Live Backtest Workbench

Phi-nance provides a modular **Live Backtest Workbench** for quantitative research:

- Fetch & cache historical data
- Select & tune indicators
- Blend multiple indicators
- PhiAI auto-tuning
- Equities + Options backtests
- Reproducible runs with RunConfig + RunHistory

---

## Directory Structure

```
Phi-nance/
├── phi/                    # Engine modules
│   ├── data/               # Data fetching & caching
│   │   └── cache.py        # Parquet cache at data_cache/{vendor}/{symbol}/{tf}/
│   ├── blending/           # Indicator blending
│   │   └── blender.py      # Weighted Sum, Voting, Regime-Weighted
│   ├── phiai/              # PhiAI auto-tuning
│   │   └── auto_tune.py    # Grid/random search, regime conditioning
│   └── run_config.py       # RunConfig schema, RunHistory storage
│
├── app_streamlit/          # Streamlit apps
│   └── live_workbench.py   # Live Backtest Workbench UI
│
├── regime_engine/          # MFT Regime Engine (existing)
│   ├── data_fetcher.py     # Alpha Vantage
│   ├── indicator_library.py
│   ├── scanner.py
│   └── ...
│
├── strategies/             # Lumibot strategies (existing)
│   ├── alpha_vantage_fixed.py
│   ├── rsi.py, macd.py, bollinger.py, ...
│   └── blended_mft_strategy.py
│
├── data_cache/             # Cached OHLCV datasets (parquet)
├── runs/                   # Run history (config.json, results.json, trades.csv)
└── .streamlit/
    ├── config.toml         # Dark theme (purple/orange)
    └── styles.css          # Custom CSS
```

---

## Data Flow

1. **Dataset Builder** → `phi.data.fetch_and_cache()` → `data_cache/{vendor}/{symbol}/{timeframe}/`
2. **Indicator Selection** → Strategy params from `INDICATOR_CATALOG`
3. **Blending** → `phi.blending.blend_signals()` (weighted_sum, voting, etc.)
4. **Backtest** → Lumibot `run_backtest()` with `AlphaVantageFixedDataSource`
5. **Run History** → `phi.run_config.RunHistory` → `runs/{run_id}/`

---

## Key Schemas

### RunConfig

- `dataset_id`, `symbols`, `start_date`, `end_date`, `timeframe`, `vendor`
- `initial_capital` (required, > 0)
- `trading_mode`: equities | options
- `indicators`, `blend_method`, `blend_weights`
- `phiai_enabled`, `phiai_constraints`
- `exit_rules`, `position_sizing`
- `evaluation_metric`

### Cache Layout

```
data_cache/
  alphavantage/
    SPY/
      1D/
        20200101_20241231.parquet
        20200101_20241231.parquet.metadata.json
```

---

## Entry Points

| Command | Purpose |
|---------|---------|
| `streamlit run app_streamlit/live_workbench.py` | Live Backtest Workbench |
| `streamlit run dashboard.py` | Legacy MFT Dashboard (6 tabs) |
| `python run_backtest.py --strategy rsi --start 2020-01-01` | CLI backtest |

---

## Theme

- **Base**: Dark
- **Primary**: `#a855f7` (purple)
- **Accent**: `#f97316` (orange)
- **Background**: `#0f0f12`, `#1a1a1f`

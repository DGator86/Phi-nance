# Phi-nance Architecture

> Version 1.0.0 — Last updated 2026-03-03

## Overview

Phi-nance is a modular, open-source quantitative research platform structured as a
top-level Python package (`phinance/`) with a Streamlit front-end (`frontend/`).

The platform follows a clean **6-step pipeline**:

```
Data Acquisition → Indicator Signals → Signal Blending → PhiAI Optimisation
       → Backtest Engine → Storage & Review
```

---

## Package Tree

```
phinance/
├── __init__.py              Top-level convenience re-exports
│
├── data/                    Data acquisition & cache layer
│   ├── cache.py             Parquet-based OHLCV cache (DataCache)
│   ├── utils.py             Resample, gap-fill, clip helpers
│   └── vendors/
│       ├── base.py          Abstract BaseVendor
│       ├── yfinance.py      yfinance adapter (daily + intraday)
│       ├── alphavantage.py  Alpha Vantage adapter
│       └── binance.py       Binance public kline adapter
│
├── strategies/              Technical-indicator library
│   ├── base.py              Abstract BaseIndicator
│   ├── indicator_catalog.py Registry + compute_indicator()
│   ├── params.py            Default & optimisation parameter grids
│   ├── rsi.py               Wilder RSI
│   ├── macd.py              MACD histogram
│   ├── bollinger.py         Bollinger Bands
│   ├── dual_sma.py          Dual SMA crossover
│   ├── ema.py               Dual EMA crossover (faster)
│   ├── mean_reversion.py    Z-score mean-reversion
│   ├── breakout.py          Donchian Channel breakout
│   ├── buy_hold.py          Buy-and-hold baseline
│   ├── vwap.py              VWAP deviation
│   ├── atr.py               ATR volatility signal
│   ├── stochastic.py        Stochastic %D
│   ├── williams_r.py        Williams %R
│   ├── cci.py               Commodity Channel Index
│   ├── obv.py               On-Balance Volume
│   └── psar.py              Parabolic SAR
│
├── blending/                Signal aggregation layer
│   ├── blender.py           blend_signals() orchestrator
│   ├── methods.py           weighted_sum, voting, regime_weighted, phiai_chooses
│   ├── weights.py           normalise_weights, equal_weights, boost_weights
│   └── regime_detector.py   Market-regime classifier (7 labels)
│
├── optimization/            PhiAI parameter tuning
│   ├── phiai.py             PhiAI class + run_phiai_optimization()
│   ├── grid_search.py       grid_search, random_search, search dispatcher
│   ├── evaluators.py        direction_accuracy, sharpe_proxy, sortino_proxy
│   └── explainer.py         Human-readable change summaries
│
├── backtest/                Core simulation engine
│   ├── runner.py            run_backtest() — high-level entry point
│   ├── engine.py            simulate() — vectorised bar-by-bar loop
│   ├── metrics.py           CAGR, Sharpe, Sortino, max_drawdown, ...
│   └── models.py            BacktestResult, Trade, Position dataclasses
│
├── options/                 Options pricing & backtesting
│   ├── pricing.py           Black-Scholes call/put + implied volatility
│   ├── greeks.py            Delta, gamma, theta, vega, rho
│   ├── backtest.py          Delta-approximation options backtest
│   ├── market_data.py       MarketDataApp connector (live chain snapshots)
│   └── ai_advisor.py        OptionsAIAdvisor — LLM + rule-based fallback
│
├── storage/                 Run persistence layer
│   ├── local.py             LocalStorage — JSON/CSV file I/O
│   ├── run_history.py       RunHistory — high-level CRUD API
│   └── models.py            StoredRun dataclass
│
├── config/                  Configuration management
│   ├── run_config.py        RunConfig — reproducible run specification
│   ├── settings.py          Settings — environment-variable loader
│   └── schema.py            Validation helpers
│
├── agents/                  LLM integrations
│   └── ollama_agent.py      OllamaAgent — Ollama REST client
│
├── phibot/                  AI post-run reviewer
│   └── reviewer.py          review_backtest() — regime-aware suggestions
│
├── utils/                   Shared utilities
│   ├── logging.py           get_logger() — centralised logging
│   ├── timing.py            Timer class + timeit decorator
│   └── decorators.py        retry, log_call, validate_ohlcv
│
└── exceptions.py            Custom exception hierarchy
```

---

## Data Flow

```
                ┌─────────────────────────────────────────────────────┐
                │                  RunConfig                          │
                └────────────┬────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │      Data Layer          │  phinance.data
              │  fetch_and_cache()       │  ─ yfinance / AlphaVantage / Binance
              │  DataCache (Parquet)     │  ─ Parquet on disk
              └──────────────┬───────────┘
                             │  OHLCV DataFrame
                             ▼
              ┌──────────────────────────┐
              │  Strategies / Indicators │  phinance.strategies
              │  compute_indicator()     │  ─ 15 built-in indicators
              │  INDICATOR_CATALOG       │  ─ returns Series ∈ [-1, 1]
              └──────────────┬───────────┘
                             │  signals DataFrame
                             ▼
              ┌──────────────────────────┐
              │  Blending Engine         │  phinance.blending
              │  blend_signals()         │  ─ weighted_sum / voting /
              │  RegimeDetector          │    regime_weighted / phiai_chooses
              └──────────────┬───────────┘
                             │  composite signal Series
                   ┌─────────┴──────────┐
                   │                    │
                   ▼                    ▼
    ┌───────────────────────┐ ┌─────────────────────────┐
    │   PhiAI Optimisation  │ │   Backtest Engine        │
    │  run_phiai_optimiz.() │ │  run_backtest()          │
    │  random_search()      │ │  simulate()              │
    └───────────┬───────────┘ └──────────┬──────────────┘
                │ optimised params        │ BacktestResult
                └─────────────┬──────────┘
                              │
                              ▼
              ┌──────────────────────────┐
              │  Storage Layer           │  phinance.storage
              │  RunHistory.create_run() │  ─ JSON config + results
              │  RunHistory.save_result()│  ─ CSV trades
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  Phibot AI Review        │  phinance.phibot
              │  review_backtest()       │  ─ regime-aware tweaks
              └──────────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  Streamlit UI            │  frontend/streamlit/
              │  6-step multi-page app   │  ─ live workbench
              └──────────────────────────┘
```

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Reproducibility** | Every run is fully captured in `RunConfig` (JSON-serialisable) |
| **Modularity** | Each sub-package has a clean public API (`__all__` + `__init__.py`) |
| **Graceful degradation** | LLM agents fall back to deterministic rules when unavailable |
| **No data leakage** | Indicators only use data available at each bar; no future lookahead |
| **Type safety** | Python 3.10+ type hints throughout; `dataclasses` for models |
| **Testability** | 295+ unit + integration tests; synthetic fixtures (no network calls) |

---

## Storage Layout

```
<project_root>/
├── data_cache/                  OHLCV Parquet cache
│   └── {vendor}/{SYMBOL}/{tf}/
│       ├── {start}_{end}.parquet
│       └── {start}_{end}.meta.json
└── runs/                        Backtest run history
    └── {YYYYMMDD_HHMMSS_hex8}/
        ├── config.json
        ├── results.json
        └── trades.csv           (optional)
```

---

## Extending the Platform

### Adding a new indicator

1. Create `phinance/strategies/myindicator.py` subclassing `BaseIndicator`
2. Implement `compute(df: pd.DataFrame, **params) -> pd.Series` returning values in `[-1, 1]`
3. Register in `phinance/strategies/indicator_catalog.py`
4. Add parameter grids to `phinance/strategies/params.py`
5. Add a boost map entry in `phinance/blending/methods.py`

### Adding a new data vendor

1. Create `phinance/data/vendors/myvendor.py` subclassing `BaseVendor`
2. Implement `fetch(symbol, timeframe, start, end, **kwargs) -> pd.DataFrame`
3. Register in `phinance/data/cache.py`'s `fetch_and_cache()` dispatcher

### Adding a new blend method

1. Add the function to `phinance/blending/methods.py`
2. Register its name in `BLEND_METHODS`
3. Add dispatch in `phinance/blending/blender.py`'s `blend_signals()`

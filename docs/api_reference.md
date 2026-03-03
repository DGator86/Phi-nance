# Phi-nance API Reference

> Version 1.0.0

## Table of Contents

- [phinance.data](#phinancedata)
- [phinance.strategies](#phinancestrategies)
- [phinance.blending](#phinanceblending)
- [phinance.optimization](#phinanceoptimization)
- [phinance.backtest](#phinancebacktest)
- [phinance.options](#phinanceoptions)
- [phinance.storage](#phinancestorage)
- [phinance.config](#phinanceconfig)
- [phinance.agents](#phinanceagents)
- [phinance.phibot](#phinancephibot)
- [phinance.utils](#phinanceutils)
- [phinance.exceptions](#phinanceexceptions)

---

## phinance.data

### `fetch_and_cache(vendor, symbol, timeframe, start, end, api_key=None)`

Fetch OHLCV data and save to the Parquet cache. Returns cached data if available.

```python
from phinance.data import fetch_and_cache

df = fetch_and_cache(
    vendor    = "yfinance",   # "yfinance" | "alphavantage" | "binance"
    symbol    = "SPY",
    timeframe = "1D",         # "1D" | "1H" | "15m" | "5m" | "1m"
    start     = "2022-01-01",
    end       = "2024-12-31",
)
# Returns pd.DataFrame with columns: open, high, low, close, volume
# DatetimeIndex
```

### `get_cached_dataset(vendor, symbol, timeframe, start, end)`

Load from cache only (no network). Raises `CacheError` if not found.

### `list_cached_datasets()`

Return a list of metadata dicts for all cached datasets.

```python
from phinance.data import list_cached_datasets

for d in list_cached_datasets():
    print(d["vendor"], d["symbol"], d["timeframe"], d["rows"])
```

### `DataCache`

Low-level cache class. Most code should use the convenience functions above.

| Method | Description |
|---|---|
| `save(vendor, symbol, timeframe, start, end, df)` | Write DataFrame to Parquet |
| `load(vendor, symbol, timeframe, start, end)` | Load from Parquet |
| `exists(vendor, symbol, timeframe, start, end)` | Check cache hit |
| `list_datasets()` | All cached datasets |

---

## phinance.strategies

### `list_indicators()`

Return all registered indicator names.

```python
from phinance.strategies import list_indicators
print(list_indicators())
# ['RSI', 'MACD', 'Bollinger', 'Dual SMA', 'EMA Cross', 'Mean Reversion',
#  'Breakout', 'Buy & Hold', 'VWAP', 'ATR', 'Stochastic', 'Williams %R',
#  'CCI', 'OBV', 'PSAR']
```

### `compute_indicator(name, df, params=None)`

Compute a named indicator's signal from OHLCV data.

```python
from phinance.strategies import compute_indicator

sig = compute_indicator("RSI", df, {"period": 14})
# Returns pd.Series with values in [-1, 1]; +1 = buy, -1 = sell, 0 = neutral
```

### `BaseIndicator`

Abstract base class for all indicators.

| Attribute | Type | Description |
|---|---|---|
| `name` | str | Display name |
| `default_params` | dict | Default parameter values |
| `param_grid` | dict | Grid values for optimisation |

| Method | Description |
|---|---|
| `compute(df, **params)` | Abstract — returns pd.Series |
| `compute_with_defaults(df, params)` | Merges params with defaults, calls compute |
| `get_param_grid()` | Returns param_grid dict |

---

## phinance.blending

### `blend_signals(signals, weights=None, method="weighted_sum", regime_probs=None)`

Blend multiple indicator signals into one composite signal.

```python
from phinance.blending import blend_signals
import pandas as pd

signals = pd.DataFrame({
    "RSI":  rsi_signal,
    "MACD": macd_signal,
})

composite = blend_signals(
    signals      = signals,
    weights      = {"RSI": 0.6, "MACD": 0.4},
    method       = "weighted_sum",   # or "voting", "regime_weighted", "phiai_chooses"
    regime_probs = None,             # pd.DataFrame for regime_weighted
)
# Returns pd.Series ∈ [-1, 1]
```

### `BLEND_METHODS`

List of all supported blend method names:
`["weighted_sum", "voting", "regime_weighted", "phiai_chooses"]`

---

## phinance.optimization

### `run_phiai_optimization(ohlcv, indicators, max_iter_per_indicator=20, timeframe="1D")`

Auto-tune indicator parameters using concurrent random search.

```python
from phinance.optimization import run_phiai_optimization

optimized, explanation = run_phiai_optimization(
    ohlcv      = df,
    indicators = {
        "RSI":  {"enabled": True, "auto_tune": True, "params": {}},
        "MACD": {"enabled": True, "auto_tune": True, "params": {}},
    },
    max_iter_per_indicator = 20,
    timeframe              = "1D",
)
# optimized: {name: {"enabled": True, "auto_tune": False, "params": {...}}}
# explanation: human-readable string of changes made
```

### `PhiAI`

Configuration container with `explain()` method.

```python
ai = PhiAI(max_indicators=5, allow_shorts=False, risk_cap=0.02)
print(ai.explain())
```

### `grid_search(ohlcv, objective_fn, param_grid, max_iter=200)`

Exhaustive grid search. Returns `(best_params, best_score)`.

### `random_search(ohlcv, objective_fn, param_grid, max_iter=50, seed=None)`

Random parameter sampling. Returns `(best_params, best_score)`.

### `direction_accuracy(ohlcv, indicator_name, params)`

Objective function: fraction of bars where signal direction matches next-bar return.
Returns float in [0, 1] (0.5 = random).

---

## phinance.backtest

### `run_backtest(ohlcv, symbol, indicators, blend_weights, blend_method, ...)`

High-level backtest entry point.

```python
from phinance.backtest import run_backtest

result = run_backtest(
    ohlcv             = df,
    symbol            = "SPY",
    indicators        = {
        "RSI":  {"enabled": True, "params": {"period": 14}},
        "MACD": {"enabled": True, "params": {}},
    },
    blend_weights     = {"RSI": 0.6, "MACD": 0.4},
    blend_method      = "weighted_sum",
    signal_threshold  = 0.15,
    initial_capital   = 100_000.0,
    position_size_pct = 0.95,
)
```

### `BacktestResult`

| Field | Type | Description |
|---|---|---|
| `symbol` | str | Ticker |
| `total_return` | float | Fractional total return |
| `cagr` | float | Annualised compound return |
| `max_drawdown` | float | Maximum peak-to-trough drawdown |
| `sharpe` | float | Sharpe ratio |
| `sortino` | float | Sortino ratio |
| `win_rate` | float | Fraction of winning trades |
| `total_trades` | int | Total closed trades |
| `portfolio_value` | list[float] | NAV per bar |
| `net_pl` | float | Total net P&L in $ |
| `trades` | list[Trade] | All closed trades |
| `prediction_log` | list[dict] | Bar-by-bar signal log |
| `metadata` | dict | Blend method, indicators, etc. |
| `to_dict()` | method | JSON-serialisable dict |

### `Trade`

| Field | Type |
|---|---|
| `entry_date` | datetime |
| `exit_date` | datetime |
| `symbol` | str |
| `entry_price` | float |
| `exit_price` | float |
| `quantity` | int |
| `pnl` | float |
| `pnl_pct` | float |
| `hold_bars` | int |
| `direction` | str |
| `regime` | str |
| `win` (property) | bool |

---

## phinance.options

### `black_scholes_call(S, K, T, r, sigma)` / `black_scholes_put(...)`

European option pricing under Black-Scholes.

```python
from phinance.options.pricing import black_scholes_call, implied_volatility

price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
iv = implied_volatility(price, S=100, K=100, T=1.0, r=0.05, option_type="call")
```

### `compute_greeks(S, K, T, r, sigma, option_type="call")`

Returns `OptionsGreeks(delta, gamma, theta, vega, rho)`.

### `run_options_backtest(ohlcv, symbol, strategy_type, initial_capital, ...)`

Delta-based options backtest. Returns dict with `portfolio_value`, `total_return`,
`cagr`, `max_drawdown`, `sharpe`, `trades`.

---

## phinance.storage

### `RunHistory`

```python
from phinance.storage import RunHistory
from phinance.config.run_config import RunConfig

history = RunHistory()                    # defaults to ./runs/
run_id  = history.create_run(cfg)         # persist config, return run_id
history.save_results(run_id, results, trades_df)
run     = history.load_run(run_id)        # returns StoredRun
runs    = history.list_runs(limit=20)     # list dicts newest-first
```

### `StoredRun`

| Property / Field | Type | Description |
|---|---|---|
| `run_id` | str | Unique run identifier |
| `config` | dict | Serialised RunConfig |
| `results` | dict | Backtest metrics |
| `trades` | pd.DataFrame or None | Closed trades |
| `symbols` (property) | list[str] | From config |
| `total_return` (property) | float | |
| `sharpe` (property) | float | |
| `cagr` (property) | float | |
| `max_drawdown` (property) | float | |
| `summary()` | str | One-line summary |

---

## phinance.config

### `RunConfig`

Fully reproducible backtest specification.

```python
from phinance.config.run_config import RunConfig

cfg = RunConfig(
    symbols         = ["SPY"],
    start_date      = "2022-01-01",
    end_date        = "2024-12-31",
    timeframe       = "1D",
    vendor          = "yfinance",
    initial_capital = 100_000.0,
    trading_mode    = "equities",
    indicators      = {"RSI": {"enabled": True, "params": {}}},
    blend_method    = "weighted_sum",
    phiai_enabled   = False,
)
cfg.validate()         # raises ConfigurationError on invalid config
d = cfg.to_dict()      # JSON-serialisable
cfg2 = RunConfig.from_dict(d)  # round-trip
```

### `get_settings()`

```python
from phinance.config.settings import get_settings

s = get_settings()
print(s.av_api_key)         # from AV_API_KEY env var
print(s.ollama_host)        # from OLLAMA_HOST env var
print(s.data_cache_dir)     # Path to data cache
```

---

## phinance.agents

### `OllamaAgent`

```python
from phinance.agents import OllamaAgent, check_ollama_ready

if check_ollama_ready():
    agent = OllamaAgent(model="llama3.2")
    reply = agent.chat("Describe a TREND_UP market regime.")
    print(reply)
```

---

## phinance.phibot

### `review_backtest(ohlcv, results, prediction_log, indicators, blend_weights, blend_method, config)`

Generate a post-run AI review.

```python
from phinance.phibot.reviewer import review_backtest

review = review_backtest(
    ohlcv          = df,
    results        = result.to_dict(),
    prediction_log = result.prediction_log,
    indicators     = active_indicators,
    blend_weights  = weights,
    blend_method   = "weighted_sum",
    config         = {"symbols": ["SPY"]},
)

print(review.verdict)       # "strong" | "moderate" | "weak" | "neutral"
print(review.summary)       # narrative summary
for obs in review.observations:
    print(f"  - {obs}")
for tweak in review.tweaks:
    print(f"  [{tweak.confidence}] {tweak.title}: {tweak.rationale}")
```

### `BacktestReview`

| Field | Type | Description |
|---|---|---|
| `summary` | str | Narrative verdict summary |
| `verdict` | str | "strong" / "moderate" / "weak" / "neutral" |
| `regime_stats` | dict | Stats per regime |
| `observations` | list[str] | Markdown-formatted observations |
| `tweaks` | list[Tweak] | Actionable suggestions |
| `total_trades` | int | |
| `win_rate` | float | |
| `avg_hold_bars` | float | |

---

## phinance.utils

### `get_logger(name)`

```python
from phinance.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("hello %s", "world")
```

### `Timer`

```python
from phinance.utils.timing import Timer
with Timer() as t:
    ...
print(f"Elapsed: {t.elapsed:.3f}s")
```

### `@retry(max_attempts, delay, exceptions)`

Retry decorator for unreliable network calls.

```python
from phinance.utils.decorators import retry

@retry(max_attempts=3, delay=2.0, exceptions=(ConnectionError,))
def fetch_data(): ...
```

---

## phinance.exceptions

| Exception | Raised by |
|---|---|
| `PhinanceError` | Base class for all errors |
| `DataFetchError` | Vendor fetch failures |
| `CacheError` | Cache read/write failures |
| `UnsupportedTimeframeError` | Vendor doesn't support timeframe |
| `IndicatorComputationError` | Indicator compute() raises |
| `UnknownIndicatorError` | Indicator not in catalog |
| `UnsupportedBlendMethodError` | Blend method not in BLEND_METHODS |
| `ConfigurationError` | Invalid RunConfig |
| `RunNotFoundError` | Run ID not in storage |
| `OptimizationError` | PhiAI search failure |

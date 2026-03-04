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
#  'CCI', 'OBV', 'PSAR',
#  'Aroon', 'Ulcer Index', 'KST', 'TRIX', 'Mass Index']
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

### Registered Indicators (20 total)

#### Original 15 Indicators

| Name | Class | Category | Default Params |
|---|---|---|---|
| `RSI` | `RSIIndicator` | Mean-reversion | period=14, oversold=30, overbought=70 |
| `MACD` | `MACDIndicator` | Momentum | fast=12, slow=26, signal=9 |
| `Bollinger` | `BollingerIndicator` | Mean-reversion | period=20, num_std=2.0 |
| `Dual SMA` | `DualSMAIndicator` | Trend | fast=50, slow=200 |
| `EMA Cross` | `EMACrossIndicator` | Trend | fast=12, slow=26 |
| `Mean Reversion` | `MeanReversionIndicator` | Mean-reversion | period=20, z_threshold=2.0 |
| `Breakout` | `BreakoutIndicator` | Breakout | period=20 |
| `Buy & Hold` | `BuyHoldIndicator` | Benchmark | — |
| `VWAP` | `VWAPIndicator` | Volume | period=20, band_pct=0.5 |
| `ATR` | `ATRIndicator` | Volatility | period=14 |
| `Stochastic` | `StochasticIndicator` | Mean-reversion | k=14, d=3, smooth=3 |
| `Williams %R` | `WilliamsRIndicator` | Mean-reversion | period=14 |
| `CCI` | `CCIIndicator` | Momentum | period=14, scale=100 |
| `OBV` | `OBVIndicator` | Volume | period=14 |
| `PSAR` | `PSARIndicator` | Trend | initial_af=0.02, max_af=0.2 |

#### Advanced Indicators (Added)

| Name | Class | Category | Signal Convention |
|---|---|---|---|
| `Aroon` | `AroonIndicator` | Trend strength | +1 = strong uptrend (Aroon Up=100, Down=0) |
| `Ulcer Index` | `UlcerIndexIndicator` | Risk/Drawdown | −1 = high drawdown risk, +1 = calm market |
| `KST` | `KSTIndicator` | Momentum | +1 = strong positive multi-period momentum |
| `TRIX` | `TRIXIndicator` | Momentum | +1 = rising triple-smoothed EMA (uptrend) |
| `Mass Index` | `MassIndexIndicator` | Reversal | −1 = reversal bulge (range expansion) |

#### `AroonIndicator`

Tushar Chande's Aroon oscillator measures how recently the highest high and
lowest low occurred within a rolling window. The oscillator (Aroon Up − Aroon
Down) spans [−100, +100], normalised to [−1, +1].

```python
from phinance.strategies.aroon import AroonIndicator

indicator = AroonIndicator()
sig = indicator.compute(df, period=25)
# +1 → new high just occurred (strong uptrend)
# −1 → new low just occurred (strong downtrend)
```

**Formula:**
```
Aroon Up   = ((period − days_since_period_high) / period) × 100
Aroon Down = ((period − days_since_period_low)  / period) × 100
Oscillator = Aroon Up − Aroon Down   ∈ [−100, +100]
signal     = Oscillator / 100
```

**Default params:** `period=25`  
**Param grid:** `period: [10, 14, 20, 25, 30]`  
**References:** Chande (1995); StockCharts ChartSchool; Stock.Indicators (.NET) `GetAroon()`

---

#### `UlcerIndexIndicator`

Peter Martin's Ulcer Index quantifies downside risk by computing the RMS of
percentage drawdowns from rolling highs. High UI = high drawdown risk = negative
signal; calm market (low UI) = near-zero signal.

```python
from phinance.strategies.ulcer_index import UlcerIndexIndicator

indicator = UlcerIndexIndicator()
sig = indicator.compute(df, period=14)
# −1 → high drawdown / high risk environment
# 0  → typical risk level
```

**Formula:**
```
rolling_max   = max(close, window=period)
drawdown_pct  = (close − rolling_max) / rolling_max × 100   [≤ 0]
UI            = sqrt(mean(drawdown_pct², window=period))
signal        = normalize(−UI)   [inverted: high UI → negative signal]
```

**Default params:** `period=14`  
**Param grid:** `period: [7, 10, 14, 20, 28]`  
**References:** Martin & McCann (1989); Stock.Indicators (.NET) `GetUlcerIndex()`

---

#### `KSTIndicator`

Martin Pring's Know Sure Thing (KST) combines four smoothed Rate-of-Change
(ROC) components at different timeframes into a single momentum oscillator.
Positive KST indicates broad-based bullish momentum across all four timeframes.

```python
from phinance.strategies.kst import KSTIndicator

indicator = KSTIndicator()
sig = indicator.compute(df)
# +1 → KST strongly positive (momentum aligned bullish across all periods)
# −1 → KST strongly negative (momentum aligned bearish)
```

**Formula:**
```
ROC(n)  = (close / close.shift(n) − 1) × 100
RCMA1   = SMA(ROC(roc1), sma1)   × 1
RCMA2   = SMA(ROC(roc2), sma2)   × 2
RCMA3   = SMA(ROC(roc3), sma3)   × 3
RCMA4   = SMA(ROC(roc4), sma4)   × 4
KST     = RCMA1 + RCMA2 + RCMA3 + RCMA4
signal  = normalize(KST)
```

**Default params:** `roc1=10, roc2=15, roc3=20, roc4=30, sma1=10, sma2=10, sma3=10, sma4=15, signal=9`  
**Param grid:** `roc1: [8,10,12], roc4: [25,30,35], signal: [7,9,12]`  
**References:** Pring (1992); Investopedia; Stock.Indicators (.NET) `GetKst()`

---

#### `TRIXIndicator`

Jack Hutson's TRIX is the percentage rate-of-change of a triple-smoothed
(three-pass) EMA. The triple smoothing acts as a low-pass filter eliminating
short cycles and noise, leaving only the dominant trend direction.

```python
from phinance.strategies.trix import TRIXIndicator

indicator = TRIXIndicator()
sig = indicator.compute(df, period=15)
# +1 → TRIX positive (triple-smoothed upward momentum)
# −1 → TRIX negative (triple-smoothed downward momentum)
```

**Formula:**
```
EMA1 = EMA(close, period)
EMA2 = EMA(EMA1,  period)
EMA3 = EMA(EMA2,  period)
TRIX = (EMA3 − EMA3.shift(1)) / EMA3.shift(1) × 100
signal = normalize(TRIX)
```

**Default params:** `period=15, signal=9`  
**Param grid:** `period: [8,10,12,15,18,21], signal: [7,9,12]`  
**References:** Hutson (1983); LuxAlgo TRIX guide; Stock.Indicators (.NET) `GetTrix()`; `python.stockindicators.dev/indicators/Trix/`

---

#### `MassIndexIndicator`

Donald Dorsey's Mass Index detects trend reversals by measuring the ratio of a
single EMA to a double EMA of the high-low range. When this ratio-sum rises
above the "reversal bulge" threshold (27) and falls back below 26.5, a trend
reversal is signalled. The MI is scale-independent (normalised as a ratio).

```python
from phinance.strategies.mass_index import MassIndexIndicator

indicator = MassIndexIndicator()
sig = indicator.compute(df, fast_period=9, slow_period=25)
# −1 → reversal bulge forming (range expansion, high MI)
# +1 → range compressed (low MI, quiet market)
```

**Formula:**
```
Single EMA  = EMA(high − low, fast_period)
Double EMA  = EMA(Single EMA, fast_period)
EMA Ratio   = Single EMA / Double EMA
Mass Index  = SUM(EMA Ratio, slow_period)
signal      = −normalize_zscore(Mass Index)   [inverted: high MI → negative]
```

**Default params:** `fast_period=9, slow_period=25, bulge_high=27.0, bulge_low=26.5`  
**Param grid:** `fast_period: [7,9,12], slow_period: [20,25,30]`  
**References:** Dorsey (1992); StockCharts ChartSchool; Wikipedia Mass Index; Stock.Indicators (.NET) `GetMassIndex()`

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

---

## phinance.options.iv_surface

Implied Volatility Surface — build, interpolate, smile, and term-structure.

### `build_iv_surface(quotes, spot, r=0.05, as_of=None, reference_date=None)`

Build an `IVSurface` from a DataFrame of option quotes.

**Required DataFrame columns:** `strike`, `expiry`, `option_type`, `bid`, `ask`

```python
from phinance.options.iv_surface import build_iv_surface

surf = build_iv_surface(quotes_df, spot=450.0, r=0.05,
                        reference_date=date(2026, 3, 3))
```

### `IVSurface` methods

| Method | Returns | Description |
|---|---|---|
| `expiries()` | `list[str]` | Sorted expiry date strings |
| `strikes()` | `list[float]` | Sorted strike prices |
| `smile_for_expiry(expiry)` | `pd.Series` | Strike → IV for one expiry |
| `term_structure(moneyness=1.0)` | `pd.Series` | ATM IV per expiry |
| `interpolate(strike, T, method)` | `float` | Bilinear / nearest IV interpolation |
| `to_dataframe()` | `pd.DataFrame` | All IVPoints as tidy table |

### `interpolate_iv(surface, strike, T, method="bilinear")`

Wrapper for `surface.interpolate()`.

### `smile_for_expiry(surface, expiry)` / `term_structure(surface, moneyness=1.0)`

Convenience wrappers for surface methods.

---

## phinance.agents

Agentic AI layer — deterministic and LLM-backed agents with orchestration.

### `AgentBase` (abstract)

```python
from phinance.agents.base import AgentBase, AgentResult

class MyAgent(AgentBase):
    @property
    def name(self) -> str:
        return "MyAgent"

    def analyze(self, context: dict) -> AgentResult:
        signal = context.get("signal", 0.0)
        action = "buy" if signal > 0.3 else "sell" if signal < -0.3 else "hold"
        return AgentResult(agent=self.name, action=action, confidence=abs(signal),
                           rationale=f"Signal: {signal:.3f}")
```

#### `AgentResult` attributes

| Field | Type | Description |
|---|---|---|
| `agent` | str | Agent name |
| `action` | str | `"buy"` \| `"sell"` \| `"hold"` |
| `confidence` | float | 0.0–1.0 |
| `rationale` | str | Human-readable explanation |
| `data` | dict | Optional payload (regime, signal, etc.) |
| `signal_value` | float (property) | action × confidence in [-1, 1] |

### `RuleBasedAgent`

Fast, deterministic agent. No LLM or network required.

```python
from phinance.agents import RuleBasedAgent
agent = RuleBasedAgent(buy_threshold=0.3, sell_threshold=-0.3, regime_boost=0.15)
result = agent.analyze({"signal": 0.7, "regime": "TREND_UP"})
# result.action → "buy", result.confidence → 0.85
```

### `AgentOrchestrator`

Multi-agent pipeline with regime detection, signal blending, and optional backtest oversight.

```python
from phinance.agents import AgentOrchestrator, RuleBasedAgent

orch = AgentOrchestrator(
    agents=[RuleBasedAgent()],
    blend_method="weighted_sum",
    backtest_fn=my_backtest_fn,   # optional: (ohlcv) → dict
)
result = orch.run(
    ohlcv_df,
    indicators={"RSI": {"enabled": True, "params": {}}},
)
print(result.consensus_action)   # "buy" / "sell" / "hold"
print(result.summary)            # human-readable pipeline summary
```

#### `OrchestratorResult` attributes

| Field | Type | Description |
|---|---|---|
| `consensus_action` | str | Aggregated vote |
| `consensus_conf` | float | Confidence-weighted average |
| `regime` | str | Detected market regime |
| `composite_signal` | float | Blended indicator signal |
| `agent_results` | list[AgentResult] | Per-agent details |
| `backtest_summary` | dict | Optional backtest stats |
| `summary` | str | Human-readable summary paragraph |
| `elapsed_ms` | float | Pipeline wall-clock time |

### `run_with_agents(ohlcv, agents=None, indicators=None, weights=None, ...)`

One-shot convenience wrapper:

```python
from phinance.agents import run_with_agents
result = run_with_agents(ohlcv_df, indicators={"RSI": {"enabled": True, "params": {}}})
```

### `OllamaAgent`

Local LLM agent via Ollama:

```python
from phinance.agents import OllamaAgent, check_ollama_ready
if check_ollama_ready():
    agent = OllamaAgent(model="llama3.2")
    reply = agent.chat("Analyse: regime TREND_UP, signal +0.7")
```

---

## phinance.blending (refactored)

The blending engine is split into 4 focused modules:

| Module | Responsibility |
|---|---|
| `blender.py` | Public `blend_signals()` dispatcher |
| `methods.py` | Pure blend implementations (weighted_sum, voting, regime_weighted) |
| `regime_detector.py` | `detect_regime()` OHLCV → label Series, `regime_to_probs()` |
| `weights.py` | `normalise_weights`, `equal_weights`, `boost_weights`, `regime_adjusted_weights` |

### `blend_signals(signals, weights=None, method="weighted_sum", regime_probs=None)`

```python
from phinance.blending import blend_signals

composite = blend_signals(
    signals_df,
    weights={"RSI": 0.4, "MACD": 0.6},
    method="regime_weighted",
    regime_probs=regime_probs_df,
)
```

### `detect_regime(ohlcv, lookback=20)`

Returns a `pd.Series` of regime labels: `TREND_UP`, `TREND_DN`, `RANGE`, `HIGHVOL`, `LOWVOL`, `BREAKOUT_UP`, `BREAKOUT_DN`.

```python
from phinance.blending.regime_detector import detect_regime, regime_to_probs
labels = detect_regime(ohlcv_df)
probs  = regime_to_probs(labels)   # one-hot probability DataFrame
```

### Blend methods

| Method | Description |
|---|---|
| `weighted_sum` | Linear weighted average |
| `voting` | Majority vote with optional threshold |
| `regime_weighted` | Regime-aware boost/dampen using `REGIME_INDICATOR_BOOST` |
| `phiai_chooses` | Placeholder → delegates to weighted_sum |

---

## phinance.optimization.phiai

### `PhiAI` config class

```python
from phinance.optimization.phiai import PhiAI, run_phiai_optimization

config = PhiAI(max_indicators=5, allow_shorts=False, risk_cap=0.02)

optimized, explanation = run_phiai_optimization(
    ohlcv_df,
    indicators={"RSI": {"enabled": True, "auto_tune": True, "params": {}}},
    max_iter_per_indicator=20,
    timeframe="1D",
)
print(config.explain())
```

### `run_phiai_optimization(ohlcv, indicators, max_iter_per_indicator=20, timeframe="1D")`

Returns `(optimized_indicators: dict, explanation: str)`.
Uses `ThreadPoolExecutor` for parallel random search across indicators.

---

## phinance.strategies — New Indicators (Phase 7)

### Moving Average Family

#### `DEMAIndicator` — Double Exponential Moving Average

```python
from phinance.strategies.dema import DEMAIndicator
ind = DEMAIndicator()
signal = ind.compute(ohlcv_df, period=21)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 21 | EMA window |

**Formula**: `DEMA = 2 × EMA(close, n) − EMA(EMA(close, n), n)`. Halves EMA lag. Signal = `(close − DEMA) / DEMA`, normalised to [−1, +1].

**Reference**: Mulloy (1994), *Technical Analysis of Stocks & Commodities*; Stock.Indicators `.NET` `GetDema()`.

---

#### `TEMAIndicator` — Triple Exponential Moving Average

```python
from phinance.strategies.tema import TEMAIndicator
signal = TEMAIndicator().compute(ohlcv_df, period=21)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 21 | EMA window |

**Formula**: `TEMA = 3×EMA1 − 3×EMA2 + EMA3`. Near-zero lag. Signal = `(close − TEMA) / TEMA`.

**Reference**: Mulloy (1994); TA-Lib TEMA; Stock.Indicators `.NET` `GetTema()`.

---

#### `KAMAIndicator` — Kaufman Adaptive Moving Average

```python
from phinance.strategies.kama import KAMAIndicator
signal = KAMAIndicator().compute(ohlcv_df, er_period=10, fast_period=2, slow_period=30)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `er_period` | int | 10 | Efficiency Ratio look-back |
| `fast_period` | int | 2 | Fast EMA period (trending) |
| `slow_period` | int | 30 | Slow EMA period (ranging) |

**Formula**: Adapts smoothing constant via Efficiency Ratio — fast in trends, slow in noise. Signal = `(close − KAMA) / KAMA`.

**Reference**: Perry Kaufman (1998), *Trading Systems and Methods*; Stock.Indicators `.NET` `GetKama()`.

---

#### `ZLEMAIndicator` — Zero Lag EMA

```python
from phinance.strategies.zlema import ZLEMAIndicator
signal = ZLEMAIndicator().compute(ohlcv_df, period=21)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 21 | EMA window |

**Formula**: `lag = (period − 1) // 2`; `ZLEMA = EMA(2 × close − close.shift(lag), period)`. Signal = `(close − ZLEMA) / ZLEMA`.

**Reference**: Ehlers & Way (2010), *Stocks & Commodities* Jan 2010; Stock.Indicators `.NET` `GetZlEma()`.

---

#### `HMAIndicator` — Hull Moving Average

```python
from phinance.strategies.hma import HMAIndicator
signal = HMAIndicator().compute(ohlcv_df, period=20)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | HMA window |

**Formula**: `HMA = WMA(2×WMA(close, n/2) − WMA(close, n), √n)`. Virtually eliminates lag. Signal = `(close − HMA) / HMA`.

**Reference**: Alan Hull (2005) [alanhull.com]; Stock.Indicators `.NET` `GetHma()`.

---

#### `VWMAIndicator` — Volume Weighted Moving Average

```python
from phinance.strategies.vwma import VWMAIndicator
signal = VWMAIndicator().compute(ohlcv_df, period=20)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | Rolling window in bars |

**Formula**: `VWMA = Σ(close × volume) / Σ(volume)` over rolling window. Signal = `(close − VWMA) / VWMA`.

**Reference**: TradingView built-in `vwma()`; Stock.Indicators `.NET` `GetVwma()`.

---

### Channel / Band Indicators

#### `IchimokuIndicator` — Ichimoku Kinko Hyo

```python
from phinance.strategies.ichimoku import IchimokuIndicator
signal = IchimokuIndicator().compute(ohlcv_df, fast_period=9, slow_period=26)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fast_period` | int | 9 | Tenkan-sen period |
| `slow_period` | int | 26 | Kijun-sen period |
| `cloud_period` | int | 26 | Senkou displacement |

**Signal**: Composite vote from price vs Tenkan (+0.4), price vs Kijun (+0.4), Tenkan vs Kijun (+0.2). Range [−1, +1].

**Reference**: Hosoda (1969); [Investopedia Ichimoku Cloud](https://www.investopedia.com/terms/i/ichimoku-cloud.asp).

---

#### `DonchianIndicator` — Donchian Channel

```python
from phinance.strategies.donchian import DonchianIndicator
signal = DonchianIndicator().compute(ohlcv_df, period=20)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | Highest-high / lowest-low window |

**Formula**: `position = (close − midpoint) / (channel_width / 2)`, already in [−1, +1]. +1 = breakout up, −1 = breakout down.

**Reference**: Richard Donchian (1960s); Stock.Indicators `.NET` `GetDonchian()`.

---

#### `KeltnerIndicator` — Keltner Channel

```python
from phinance.strategies.keltner import KeltnerIndicator
signal = KeltnerIndicator().compute(ohlcv_df, period=20, multiplier=2.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | EMA and ATR period |
| `multiplier` | float | 2.0 | ATR band multiplier |

**Formula**: `middle = EMA(close, n)`, `ATR` via Wilder's smoothing, `position = (close − middle) / (multiplier × ATR)`, clipped to [−1, +1].

**Reference**: Keltner (1960) / Raschke ATR variant; Stock.Indicators `.NET` `GetKeltner()`.

---

### Oscillators

#### `ElderRayIndicator` — Elder Ray Index

```python
from phinance.strategies.elder_ray import ElderRayIndicator
signal = ElderRayIndicator().compute(ohlcv_df, period=13)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 13 | EMA period |

**Formula**: `Bull Power = high − EMA`, `Bear Power = low − EMA`, `Net = Bull + Bear`. Normalised to [−1, +1].

**Reference**: Dr. Alexander Elder (1993), *Trading for a Living*; Stock.Indicators `.NET` `GetElderRay()`.

---

#### `DPOIndicator` — Detrended Price Oscillator

```python
from phinance.strategies.dpo import DPOIndicator
signal = DPOIndicator().compute(ohlcv_df, period=20)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | SMA period |

**Formula**: `lookback = period // 2 + 1`; `DPO = close.shift(lookback) − SMA(close, period)`. Contrarian signal — negative DPO (cycle trough) maps to +1 (buy).

**Reference**: [StockCharts DPO](https://chartschool.stockcharts.com); Stock.Indicators `.NET` `GetDpo()`.

---

### Indicator Catalog Summary (31 total)

| # | Name | Category | Key Params |
|---|---|---|---|
| 1 | RSI | Mean-reversion | period, oversold, overbought |
| 2 | MACD | Momentum | fast, slow, signal |
| 3 | Bollinger | Mean-reversion | period, num_std |
| 4 | Dual SMA | Trend | fast_period, slow_period |
| 5 | EMA Cross | Trend | fast_period, slow_period |
| 6 | Mean Reversion | Mean-reversion | period, z_threshold |
| 7 | Breakout | Breakout | period |
| 8 | Buy & Hold | Benchmark | — |
| 9 | VWAP | Volume | period, band_pct |
| 10 | ATR | Volatility | period, lookback, z_threshold |
| 11 | Stochastic | Mean-reversion | k_period, d_period, smooth |
| 12 | Williams %R | Mean-reversion | period, oversold, overbought |
| 13 | CCI | Oscillator | period, scale |
| 14 | OBV | Volume/momentum | period |
| 15 | PSAR | Trend | initial_af, step_af, max_af |
| 16 | Aroon | Trend strength | period |
| 17 | Ulcer Index | Drawdown/risk | period |
| 18 | KST | Multi-period momentum | roc1..roc4, signal |
| 19 | TRIX | Momentum | period, signal |
| 20 | Mass Index | Reversal | fast_period, slow_period |
| 21 | DEMA | Trend (low-lag MA) | period |
| 22 | TEMA | Trend (ultra low-lag MA) | period |
| 23 | KAMA | Adaptive trend | er_period, fast_period, slow_period |
| 24 | ZLEMA | Trend (zero-lag MA) | period |
| 25 | HMA | Trend (hull MA) | period |
| 26 | VWMA | Volume-weighted trend | period |
| 27 | Ichimoku | Trend/momentum | fast_period, slow_period, cloud_period |
| 28 | Donchian | Breakout/channel | period |
| 29 | Keltner | Volatility/breakout | period, multiplier |
| 30 | Elder Ray | Bull/bear power | period |
| 31 | DPO | Cycle/detrend | period |

---

## Phase 8: Platform Enhancements

---

## Docker Deployment

Phinance ships with a complete Docker-based deployment stack.

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage production image (builder + runtime) |
| `docker-compose.yml` | Orchestrates `app` + `nginx` services |
| `docker/nginx/nginx.conf` | Reverse-proxy config for Streamlit on port 80/443 |
| `.dockerignore` | Excludes caches, secrets, and data blobs |

### Quick Start

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your API keys before building:

```bash
cp .env.example .env
# Edit AV_API_KEY, MARKETDATAAPP_API_TOKEN, etc.
docker compose up --build -d
```

### Ports

| Service | Internal | Exposed |
|---|---|---|
| Streamlit app | 8501 | via nginx:80 |
| nginx | 80/443 | 80/443 |

---

## phinance.optimization — Advanced Optimizers

### Bayesian Search (`phinance.optimization.bayesian`)

Uses **Optuna's TPE (Tree-structured Parzen Estimator)** sampler for sample-efficient hyperparameter tuning.

```python
from phinance.optimization.bayesian import bayesian_search, create_study

params, score = bayesian_search(
    ohlcv_df,
    objective_fn=my_obj,
    param_grid={"period": [7, 14, 21], "oversold": [25, 30]},
    n_trials=50,
    seed=42,
)
```

#### `bayesian_search(ohlcv, objective_fn, param_grid, n_trials=50, seed=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ohlcv` | DataFrame | — | OHLCV price data |
| `objective_fn` | callable | — | `f(df, params) → float` (higher = better) |
| `param_grid` | dict | — | `{param: [values]}` search space |
| `n_trials` | int | 50 | Number of Optuna trials |
| `seed` | int\|None | None | RNG seed for reproducibility |

**Returns**: `(best_params: dict, best_score: float)`

#### `create_study(n_trials=50, seed=None) → optuna.Study`

Creates and configures an Optuna study with the TPE sampler (multivariate mode enabled).

---

### Genetic Algorithm (`phinance.optimization.genetic`)

Implements a **steady-state genetic algorithm** with tournament selection, uniform crossover, and Gaussian mutation.

```python
from phinance.optimization.genetic import genetic_search

params, score = genetic_search(
    ohlcv_df,
    objective_fn=my_obj,
    param_grid={"period": [7, 14, 21, 28], "oversold": [25, 30, 35]},
    population_size=20,
    n_generations=30,
    seed=0,
)
```

#### `genetic_search(ohlcv, objective_fn, param_grid, population_size=20, n_generations=30, mutation_rate=0.2, seed=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ohlcv` | DataFrame | — | OHLCV price data |
| `objective_fn` | callable | — | `f(df, params) → float` |
| `param_grid` | dict | — | `{param: [values]}` |
| `population_size` | int | 20 | Number of individuals per generation |
| `n_generations` | int | 30 | Evolutionary iterations |
| `mutation_rate` | float | 0.2 | Per-gene mutation probability |
| `seed` | int\|None | None | RNG seed |

**Returns**: `(best_params: dict, best_score: float)`

**Algorithm details**:
1. Initialize random population from grid values
2. Evaluate fitness using `objective_fn`
3. Tournament selection (size=3) for parents
4. Uniform crossover to produce offspring
5. Gaussian mutation clipped to valid grid values
6. Steady-state replacement (worst individual replaced)
7. Repeat for `n_generations`

---

### Updated `search()` Dispatcher

The `phinance.optimization.grid_search.search()` function now supports four methods:

```python
from phinance.optimization.grid_search import search

params, score = search(
    ohlcv_df, objective_fn, param_grid,
    method="bayesian",   # "grid" | "random" | "bayesian" | "genetic"
    max_iter=50,
)
```

| `method` | Algorithm | Best For |
|---|---|---|
| `"grid"` | Exhaustive grid search | Small grids, complete coverage |
| `"random"` | Random sampling | Large grids, fast baseline |
| `"bayesian"` | Optuna TPE | Medium grids, sample efficiency |
| `"genetic"` | Evolutionary GA | Non-convex, complex landscapes |

#### `SEARCH_METHODS` constant

```python
from phinance.optimization.grid_search import SEARCH_METHODS
# ["grid", "random", "bayesian", "genetic"]
```

---

### `PhiAI` — Updated with `search_method`

```python
from phinance.optimization.phiai import PhiAI, run_phiai_optimization

ai = PhiAI(
    max_indicators=5,
    allow_shorts=False,
    risk_cap=0.02,
    search_method="bayesian",   # NEW: "random" | "bayesian" | "genetic"
)
print(ai.explain())

result, explanation = run_phiai_optimization(
    ohlcv_df,
    indicators={"RSI": True, "MACD": True},
    max_iter_per_indicator=30,
    timeframe="1D",
    search_method="bayesian",   # NEW parameter
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `search_method` | str | `"random"` | Optimizer: `"random"`, `"bayesian"`, `"genetic"` |

---

## phinance.strategies.ml_classifier — LightGBM Indicator

Wraps a trained **LightGBM classifier** as a `BaseIndicator`, generating buy/sell signals from engineered features.

### `LGBMClassifierIndicator`

```python
from phinance.strategies.ml_classifier import LGBMClassifierIndicator

indicator = LGBMClassifierIndicator()
signal = indicator.compute(
    ohlcv_df,
    n_estimators=200,
    num_leaves=63,
    learning_rate=0.05,
    lookback=20,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | int | 100 | Number of boosting trees |
| `num_leaves` | int | 31 | Maximum leaves per tree |
| `learning_rate` | float | 0.05 | Shrinkage rate |
| `lookback` | int | 20 | Feature window (bars) |

**Returns**: `pd.Series` of signals in `[-1.0, 1.0]` (NaN-filled for warm-up period)

### Feature Engineering

The indicator engineers the following features from raw OHLCV data:

| Feature | Description |
|---|---|
| `ret_1` | 1-bar return |
| `ret_5` | 5-bar return |
| `ret_20` | 20-bar return |
| `vol_5` | 5-bar rolling std of returns |
| `vol_20` | 20-bar rolling std of returns |
| `rsi_14` | RSI(14) |
| `macd` | MACD line (12/26) |
| `bb_pct` | Bollinger %B (20, 2σ) |
| `volume_ratio` | Volume / 20-bar SMA volume |
| `hl_range` | (High − Low) / Close |

### Training Pipeline

```python
from phinance.strategies.ml_classifier import train_lgbm_model, create_features

# 1. Create features
features_df = create_features(ohlcv_df, lookback=20)

# 2. Train model (returns fitted LGBMClassifier)
model = train_lgbm_model(
    features_df,
    target_col="forward_return",
    n_estimators=200,
    num_leaves=63,
    learning_rate=0.05,
)

# 3. Save / load
import joblib
joblib.dump(model, "lgbm_model.pkl")
model = joblib.load("lgbm_model.pkl")
```

### Catalog Registration

```python
from phinance.strategies.indicator_catalog import compute_indicator
signal = compute_indicator("LGBM Classifier", ohlcv_df, {
    "n_estimators": 200,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "lookback": 20,
})
```

---

## phinance.live — Live & Paper Trading

Provides broker-agnostic paper/live trading with Alpaca and IBKR adapters.

### Architecture

```
LiveTradingLoop
    ├── BrokerAdapter (abstract)
    │     ├── AlpacaBroker
    │     ├── IBKRBroker
    │     └── PaperBroker  (in-memory simulation)
    └── run_once(ohlcv, symbol) → TradeResult
```

### `PaperBroker`

In-memory paper trading broker for backtesting/simulation — **no API keys required**.

```python
from phinance.live.broker import PaperBroker

broker = PaperBroker(initial_capital=50_000.0)
status = broker.get_account_status()
# {"equity": 50000.0, "cash": 50000.0, "pnl": 0.0, "positions": {}}
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `place_order` | `(symbol, qty, side, price)` | Execute simulated order |
| `get_account_status` | `()` | Return equity/cash/PnL/positions dict |
| `get_positions` | `()` | Return `{symbol: qty}` dict |
| `close_position` | `(symbol, price)` | Close existing position at price |

---

### `AlpacaBroker`

Connects to the **Alpaca Markets** REST API for paper/live trading.

```python
from phinance.live.broker import AlpacaBroker

broker = AlpacaBroker(
    api_key="YOUR_ALPACA_API_KEY",
    secret_key="YOUR_ALPACA_SECRET",
    base_url="https://paper-api.alpaca.markets",   # paper trading
)
order = broker.place_order("SPY", qty=10, side="buy", price=450.0)
```

| Parameter | Description |
|---|---|
| `api_key` | Alpaca API key |
| `secret_key` | Alpaca secret key |
| `base_url` | `paper-api.alpaca.markets` (paper) or `api.alpaca.markets` (live) |

---

### `IBKRBroker`

Connects to **Interactive Brokers** via the `ibapi` client library.

```python
from phinance.live.broker import IBKRBroker

broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
broker.connect()
order = broker.place_order("SPY", qty=10, side="buy", price=450.0)
```

| Parameter | Default | Description |
|---|---|---|
| `host` | `"127.0.0.1"` | TWS/Gateway hostname |
| `port` | `7497` | TWS paper port (7497) or live port (7496) |
| `client_id` | `1` | Unique client identifier |

---

### `LiveTradingLoop`

Orchestrates data fetching → signal computation → order execution.

```python
from phinance.live.live_loop import LiveTradingLoop
from phinance.live.broker import PaperBroker

loop = LiveTradingLoop(
    broker=PaperBroker(initial_capital=50_000),
    indicators={"RSI": True, "MACD": True},
    symbol="SPY",
    signal_threshold=0.3,
    position_size_pct=0.1,
)

# Single iteration (call from scheduler / cron / event loop)
result = loop.run_once(ohlcv_df)
print(result)
# TradeResult(signal=-0.248, action="hold", pnl=0.0, ...)
```

#### `run_once(ohlcv) → TradeResult`

| Field | Description |
|---|---|
| `signal` | Composite signal value `[-1, 1]` |
| `action` | `"buy"` \| `"sell"` \| `"hold"` |
| `qty` | Shares traded (0 if hold) |
| `pnl` | Realised PnL for the bar |
| `account` | Full account status snapshot |

---

### Indicator Catalog Summary (32 total)

| # | Name | Category | Key Params |
|---|---|---|---|
| 1 | RSI | Mean-reversion | period, oversold, overbought |
| 2 | MACD | Momentum | fast, slow, signal |
| 3 | Bollinger | Mean-reversion | period, num_std |
| 4 | Dual SMA | Trend | fast_period, slow_period |
| 5 | EMA Cross | Trend | fast_period, slow_period |
| 6 | Mean Reversion | Mean-reversion | period, z_threshold |
| 7 | Breakout | Breakout | period |
| 8 | Buy & Hold | Benchmark | — |
| 9 | VWAP | Volume | period, band_pct |
| 10 | ATR | Volatility | period, lookback, z_threshold |
| 11 | Stochastic | Mean-reversion | k_period, d_period, smooth |
| 12 | Williams %R | Mean-reversion | period, oversold, overbought |
| 13 | CCI | Oscillator | period, scale |
| 14 | OBV | Volume/momentum | period |
| 15 | PSAR | Trend | initial_af, step_af, max_af |
| 16 | Aroon | Trend strength | period |
| 17 | Ulcer Index | Drawdown/risk | period |
| 18 | KST | Multi-period momentum | roc1..roc4, signal |
| 19 | TRIX | Momentum | period, signal |
| 20 | Mass Index | Reversal | fast_period, slow_period |
| 21 | DEMA | Trend (low-lag MA) | period |
| 22 | TEMA | Trend (ultra low-lag MA) | period |
| 23 | KAMA | Adaptive trend | er_period, fast_period, slow_period |
| 24 | ZLEMA | Trend (zero-lag MA) | period |
| 25 | HMA | Trend (hull MA) | period |
| 26 | VWMA | Volume-weighted trend | period |
| 27 | Ichimoku | Trend/momentum | fast_period, slow_period, cloud_period |
| 28 | Donchian | Breakout/channel | period |
| 29 | Keltner | Volatility/breakout | period, multiplier |
| 30 | Elder Ray | Bull/bear power | period |
| 31 | DPO | Cycle/detrend | period |
| 32 | LGBM Classifier | ML/classification | n_estimators, num_leaves, learning_rate, lookback |


---

## Phase 9: Agentic Autonomy, Plugin System & Vectorized Backtesting

---

## phinance.agents — Autonomous Pipeline (Phase 9.1)

Full agentic autonomy: the platform can **propose**, **validate**, and **deploy** strategies without human intervention.

### Architecture

```
AutonomousPipeline
    ├── StrategyProposerAgent   → StrategyProposal
    ├── StrategyValidator       → ValidationResult
    └── AutonomousDeployer      → DeploymentRecord
           └── StrategyRegistry   (in-memory + JSON persistence)
```

---

### `StrategyProposal`

```python
from phinance.agents.strategy_proposer import StrategyProposal
```

| Attribute | Type | Description |
|---|---|---|
| `indicators` | dict | `{name: {"enabled": True, "params": {...}}}` |
| `weights` | dict | Blend weights summing to 1.0 |
| `blend_method` | str | `"weighted_sum"` \| `"regime_weighted"` |
| `regime` | str | Detected regime (e.g. `"TREND_UP"`) |
| `scores` | dict | `{name: direction_accuracy_score}` |
| `rationale` | str | Human-readable explanation |

---

### `StrategyProposerAgent`

```python
from phinance.agents.strategy_proposer import StrategyProposerAgent

agent = StrategyProposerAgent(
    top_n=4,           # indicators per proposal
    min_score=0.50,    # minimum direction_accuracy threshold
    blend_method="weighted_sum",
    probe_bars=60,     # bars for accuracy probe
)
proposal = agent.propose(ohlcv_df)
# proposal.indicators → {"EMA Cross": {...}, "MACD": {...}, ...}
# proposal.weights    → {"EMA Cross": 0.35, "MACD": 0.33, ...}
```

**Regime affinity**: The agent automatically selects indicators compatible with the current regime — trend indicators for `TREND_UP`, oscillators for `RANGE`, volatility indicators for `HIGHVOL`, etc.

#### `propose(ohlcv) → StrategyProposal`

1. Detects regime via `detect_regime()`
2. Scores all regime-compatible indicators with `direction_accuracy()` probe
3. Selects top-N by score, falls back to affinity list if scores are low
4. Computes inverse-score blend weights

---

### `StrategyValidator`

```python
from phinance.agents.strategy_validator import StrategyValidator

validator = StrategyValidator(
    min_sharpe=0.3,      # minimum acceptable Sharpe ratio
    max_drawdown=0.30,   # maximum tolerable drawdown
    min_win_rate=0.40,   # minimum win rate
    min_trades=2,        # minimum trades required
    initial_capital=100_000,
)
result = validator.validate(proposal, ohlcv_df, symbol="SPY")
# result.approved   → True/False
# result.sharpe     → 1.24
# result.rejection_reason → "" (or "Sharpe 0.12 < 0.30")
```

#### `ValidationResult` attributes

| Attribute | Description |
|---|---|
| `approved` | `True` if all thresholds met |
| `sharpe` | Annualised Sharpe ratio |
| `max_drawdown` | Worst peak-to-trough loss |
| `win_rate` | Fraction of winning trades |
| `num_trades` | Completed round-trip trades |
| `rejection_reason` | Semicolon-delimited list of failed checks |
| `backtest_stats` | Full `BacktestResult.to_dict()` payload |

---

### `AutonomousDeployer` + `StrategyRegistry`

```python
from phinance.agents.autonomous_deployer import AutonomousDeployer, StrategyRegistry

registry = StrategyRegistry(registry_path="deployments/registry.json")
deployer = AutonomousDeployer(registry=registry, dry_run=True, max_active=3)

record = deployer.deploy(validation_result, strategy_name="my_strategy")
# record.deployment_id → "uuid-..."
# record.status        → "active"

deployer.rollback(record.deployment_id)
# record.status → "rolled_back"
```

#### `DeploymentRecord` attributes

| Attribute | Description |
|---|---|
| `deployment_id` | UUID string |
| `strategy_name` | Human label |
| `status` | `"active"` \| `"rolled_back"` \| `"failed"` |
| `dry_run` | Paper mode flag |
| `regime_at_deploy` | Market regime when deployed |
| `validation_stats` | Sharpe, DD, win rate at deployment |
| `deployed_at` | Epoch timestamp |

---

### `AutonomousPipeline`

End-to-end orchestrator: **Propose → Validate → Deploy (with retries)**.

```python
from phinance.agents.autonomous_pipeline import AutonomousPipeline, run_autonomous_pipeline

# Full control
pipeline = AutonomousPipeline(
    proposer=StrategyProposerAgent(top_n=4),
    validator=StrategyValidator(min_sharpe=0.3),
    deployer=AutonomousDeployer(dry_run=True),
    max_retries=3,
)
result = pipeline.run_once(ohlcv_df, symbol="SPY")
# result.success     → True
# result.deployment  → DeploymentRecord(...)
# result.attempts    → 2

# Run multiple iterations
results = pipeline.run_loop(ohlcv_df, n_iterations=5)

# One-shot convenience
result = run_autonomous_pipeline(
    ohlcv_df,
    symbol="SPY",
    dry_run=True,
    min_sharpe=0.3,
    top_n=4,
    registry_path="deployments/registry.json",
)
```

#### `PipelineRunResult` attributes

| Attribute | Description |
|---|---|
| `success` | `True` if strategy was deployed |
| `proposal` | `StrategyProposal` from last attempt |
| `validation` | `ValidationResult` from last attempt |
| `deployment` | `DeploymentRecord` (if deployed) |
| `attempts` | Number of propose/validate cycles |
| `total_elapsed_ms` | Wall-clock time for full run |
| `message` | Human-readable summary |

**Retry logic**: On rejection, `top_n` is decremented and `min_score` is lowered by 0.02 per retry, maximising chances of finding an approvable strategy.

---

## phinance.plugins — Plugin System (Phase 9.2)

Third-party indicators and data vendors can be registered without modifying core code.

### Quick Start

```python
from phinance.plugins import (
    register_indicator_plugin,
    register_vendor_plugin,
    list_plugins,
    get_indicator_plugin,
    get_vendor_plugin,
)
```

---

### Decorator API (recommended)

```python
from phinance.plugins import register_indicator_plugin, register_vendor_plugin
from phinance.strategies.base import BaseIndicator
import pandas as pd

@register_indicator_plugin(
    "My Custom RSI",
    metadata={"version": "1.0", "author": "Alice"},
)
class MyCustomRSI(BaseIndicator):
    name = "My Custom RSI"
    def compute(self, df: pd.DataFrame, **params) -> pd.Series:
        # ... custom logic ...
        return pd.Series(0.0, index=df.index)


@register_vendor_plugin("my_broker", metadata={"auth_required": True})
def fetch_my_broker_data(symbol: str, start: str, end: str, **kwargs) -> pd.DataFrame:
    # ... fetch from broker API ...
    return pd.DataFrame(...)
```

---

### Direct registration

```python
from phinance.plugins import register_indicator, register_vendor

register_indicator("Custom ATR Variant", MyAtrClass(), metadata={"version": "2.1"})
register_vendor("my_data_feed", fetch_fn)
```

---

### Discovery

```python
from phinance.plugins import list_plugins, get_indicator_plugin, get_vendor_plugin

plugins = list_plugins()
# {"indicators": ["My Custom RSI", ...], "vendors": ["my_broker", ...]}

ind = get_indicator_plugin("My Custom RSI")
signal = ind.compute(ohlcv_df)

vendor_fn = get_vendor_plugin("my_broker")
df = vendor_fn("SPY", "2023-01-01", "2024-01-01")
```

---

### Entry-point discovery (package integration)

Third-party packages declare plugins in `pyproject.toml`:

```toml
[project.entry-points."phinance.indicators"]
my_rsi_v2 = "my_package.indicators:MyRSIV2"

[project.entry-points."phinance.vendors"]
my_feed = "my_package.vendors:fetch_data"
```

Then auto-load at startup:

```python
from phinance.plugins import load_entry_point_plugins
stats = load_entry_point_plugins()
# {"indicators_loaded": 1, "vendors_loaded": 1}
```

---

### Directory scan

```python
from phinance.plugins import load_plugin_directory
stats = load_plugin_directory("plugins/", recursive=True)
# {"files_loaded": 5, "errors": 0}
```

Any `*.py` file in the directory that uses `@register_indicator_plugin` or `@register_vendor_plugin` will be auto-registered.

---

### `PluginRegistry` API

| Method | Description |
|---|---|
| `register_indicator(name, ind, metadata, overwrite)` | Register indicator |
| `register_vendor(name, fn, metadata, overwrite)` | Register vendor |
| `get_indicator(name)` | Retrieve indicator by name |
| `get_vendor(name)` | Retrieve vendor by name |
| `list_indicators()` | Sorted list of indicator names |
| `list_vendors()` | Sorted list of vendor names |
| `list_plugins()` | `{"indicators": [...], "vendors": [...]}` |
| `get_metadata(type, name)` | Retrieve stored metadata dict |

---

## phinance.backtest.vectorized — Vectorized Engine (Phase 9.3)

50–200× faster than the event-driven engine for large datasets, implemented as pure NumPy array operations.

### Quick Start

```python
from phinance.backtest.vectorized import run_vectorized_backtest
from phinance.strategies.rsi import RSIIndicator

signal = RSIIndicator().compute(ohlcv_df)
result = run_vectorized_backtest(
    ohlcv_df,
    signal,
    symbol="SPY",
    initial_capital=100_000,
    position_size=0.95,
    signal_threshold=0.15,
    position_style="long_only",    # "long_only" | "long_short" | "long_flat"
    transaction_cost=0.001,
)
print(result.summary())
# SPY | Return=8.23% CAGR=6.12% Sharpe=0.842 DD=12.4% WinRate=58.3% Trades=24
```

---

### `run_vectorized_backtest(ohlcv, signal, ...) → VectorizedBacktestResult`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ohlcv` | DataFrame | — | OHLCV price data |
| `signal` | Series | — | Composite signal in `[-1, 1]` |
| `symbol` | str | `""` | Ticker label |
| `initial_capital` | float | `100_000` | Starting NAV |
| `position_size` | float | `0.95` | Fraction of capital per trade |
| `signal_threshold` | float | `0.15` | Min `|signal|` for entry |
| `position_style` | str | `"long_only"` | Position mode |
| `transaction_cost` | float | `0.001` | Per-trade cost fraction |

---

### `VectorizedBacktestResult` attributes

| Attribute | Type | Description |
|---|---|---|
| `total_return` | float | Fractional return |
| `cagr` | float | Compound annual growth rate |
| `sharpe` | float | Annualised Sharpe ratio |
| `sortino` | float | Annualised Sortino ratio |
| `max_drawdown` | float | Worst peak-to-trough drawdown |
| `win_rate` | float | Fraction of profitable trades |
| `num_trades` | int | Completed round-trip trades |
| `equity_curve` | np.ndarray | NAV at each bar |
| `positions` | np.ndarray | Position vector `(+1/0/-1)` |

---

### `vectorized_positions(signal, threshold, position_style) → np.ndarray`

Converts a signal array to a position array:

| `position_style` | Long signal | Short signal | Neutral |
|---|---|---|---|
| `"long_only"` | `+1` | `0` | hold |
| `"long_short"` | `+1` | `-1` | hold |
| `"long_flat"` | `+1` | `0` | `0` |

Positions are **forward-filled** — a position is held until a counter-signal appears.

---

### `run_vectorized_batch(ohlcv, signals, **kwargs) → dict`

Run multiple signals in one call:

```python
from phinance.backtest.vectorized import run_vectorized_batch
from phinance.strategies.indicator_catalog import compute_indicator

signals = {
    name: compute_indicator(name, ohlcv_df, {})
    for name in ["RSI", "MACD", "Bollinger", "EMA Cross"]
}
results = run_vectorized_batch(ohlcv_df, signals)
# {"RSI": VectorizedBacktestResult, "MACD": ..., ...}

# Rank by Sharpe
ranked = sorted(results.items(), key=lambda kv: kv[1].sharpe, reverse=True)
```

---

### `equity_curve(closes, positions, initial_capital, position_size, transaction_cost) → np.ndarray`

Low-level equity curve computation from raw NumPy arrays. Use directly for custom simulation loops.

---

### Performance comparison

| Engine | 2000-bar dataset | 50 000-bar dataset |
|---|---|---|
| Bar-by-bar (`engine.simulate`) | ~15ms | ~400ms |
| Vectorized (`run_vectorized_backtest`) | ~1ms | ~8ms |
| Speed-up | **~15×** | **~50×** |


---

## Phase 9 — Test Suite Summary (1,654 Tests)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_agentic_autonomy_extended | 62 | StrategyProposerAgent, StrategyValidator, AutonomousDeployer, StrategyRegistry |
| test_plugin_system_extended | 41 | PluginRegistry, decorators, module helpers, load_plugin_directory |
| test_vectorized_backtest_extended | 45 | vectorized_positions, equity_curve, run_vectorized_backtest, VectorizedBacktestResult |
| test_indicators_comprehensive | 378 | All 32 catalog indicators (parametrized) |
| test_backtest_metrics_extended | 50 | bars_per_year, total_return, cagr, max_drawdown, sharpe, sortino, win_rate, Trade |
| test_live_trading_extended | 54 | PaperBroker, Order/Fill/Position, LiveTradingLoop |
| **Total new (Phase 9.4)** | **630** | |
| **Grand Total** | **1,654** | 0 failures, 4 intentional skips |

### Streamlit UI Pages Added (Phase 9.6)

- **`app_streamlit/pages/plugin_browser.py`** — Browse registered plugins, inspect metadata, load `.py` plugin files via upload or directory scan.
- **`app_streamlit/pages/live_trading_dashboard.py`** — Real-time paper trading: price updates, order entry, position tracking, equity curve, fills log.
- **`app_streamlit/pages/autonomous_pipeline_page.py`** — One-click Propose → Validate → Deploy pipeline with regime detection, strategy scoring, validation thresholds, deployment records, and rollback.

Access via the **Advanced Tools** radio selector at the bottom of the main Streamlit app.

---

## Phase 10 — Full Agentic Autonomy & Community Plugin Marketplace

### `phinance.agents.evolution_engine`

**EvolutionEngine** — genetic algorithm that continuously evolves trading strategies.

```python
from phinance.agents.evolution_engine import EvolutionEngine, EvolutionConfig, run_evolution

cfg    = EvolutionConfig(population_size=10, num_generations=5, dry_run=True)
engine = EvolutionEngine(ohlcv=df, config=cfg)
history = engine.run()          # list[GenerationResult]
best    = engine.best_individual  # Individual with highest fitness

# One-shot
history, best = run_evolution(df, population_size=8, num_generations=3)
```

| Class / Function | Description |
|---|---|
| `Individual` | One candidate strategy (indicators, weights, fitness) |
| `GenerationResult` | Summary of one generation |
| `EvolutionConfig` | Hyper-parameters (pop size, mutation rate, …) |
| `EvolutionEngine` | Main genetic algorithm controller |
| `run_evolution()` | Convenience one-shot function |
| `_fitness()` | `Sharpe × (1−DD) × √trades` |

---

### `phinance.backtest.walk_forward`

**WalkForwardHarness** — rolling in-sample / out-of-sample optimisation.

```python
from phinance.backtest.walk_forward import WalkForwardHarness, WalkForwardConfig, run_walk_forward

result = run_walk_forward(df, is_bars=120, oos_bars=60, step_bars=60)
print(result.summary())
```

| Field | Description |
|---|---|
| `WFOWindow` | One IS/OOS fold with chosen indicator and OOS metrics |
| `WFOResult` | Aggregate: combined OOS Sharpe, efficiency ratio, gate flag |
| `WalkForwardConfig` | `is_bars`, `oos_bars`, `step_bars`, `gate_threshold` |

---

### `phinance.plugins.marketplace`

**MarketplaceRegistry** — versioned, discoverable plugin marketplace.

```python
from phinance.plugins.marketplace import get_marketplace, PluginManifest, scaffold_plugin

mp = get_marketplace()

@mp.publish(PluginManifest(name="My RSI", version="1.0.0", author="Alice"))
class MyRSI(BaseIndicator):
    ...

entry   = mp.get("My RSI")          # PluginEntry
results = mp.search(tags=["momentum"])

# Generate plugin skeleton
code = scaffold_plugin("My Custom MA", author="Alice", version="0.1.0")
```

| Feature | Description |
|---|---|
| SemVer versioning | Enforces `major.minor.patch` on every plugin |
| Dependency check | Warns if `requires` packages are missing |
| Conflict detection | Raises on duplicate (name, version) unless `overwrite=True` |
| `search()` | Filter by name, type, tags |
| `scaffold_plugin()` | Generate indicator or vendor plugin skeleton |

---

### `phinance.backtest.portfolio`

**PortfolioBacktester** — multi-asset correlated backtest.

```python
from phinance.backtest.portfolio import run_portfolio_backtest

result = run_portfolio_backtest(
    ohlcv_dict={"SPY": spy_df, "QQQ": qqq_df},
    signals={"SPY": spy_sig, "QQQ": qqq_sig},
    allocation="risk_parity",
    initial_capital=200_000,
)
print(result.summary())
```

| `AllocationMethod` | Description |
|---|---|
| `EQUAL` | Equal capital to every asset |
| `RISK_PARITY` | Inverse-volatility weighting |
| `FIXED` | User-supplied weight dict |

---

### `phinance.live.scheduler`

**TradingScheduler** — cron-style autonomous paper-trading loop.

```python
from phinance.live.scheduler import TradingScheduler, SchedulerConfig, run_paper_scheduler

ticks = run_paper_scheduler(
    ohlcv_dict={"SPY": spy_df},
    indicator_name="EMA Cross",
    max_ticks=5,
    dry_run=True,
)
```

---

## Phase 10 — Test Suite Summary (1,927 Tests)

| Test File | New Tests | Coverage |
|-----------|-----------|----------|
| test_evolution_engine.py | 75 | _fitness, Individual, GenerationResult, EvolutionConfig, EvolutionEngine (init/population/evaluate/blend/select/mutate/crossover/evolve/run_once/run/summary), run_evolution |
| test_walk_forward.py | 57 | WFOWindow, WFOResult, WalkForwardConfig, WalkForwardHarness (init/windows_count/_run_window/run), aggregate metrics, run_walk_forward |
| test_plugin_marketplace.py | 69 | PluginManifest (validation/deps/repr), PluginEntry, MarketplaceRegistry (registration/conflict/overwrite/versions/search/catalogue/singleton), scaffold_plugin |
| test_portfolio_backtest.py | 56 | AllocationMethod, PortfolioConfig, AssetResult, PortfolioResult, PortfolioBacktester (init/allocations/simulate/aggregate/correlation/run), run_portfolio_backtest |
| test_trading_scheduler.py | 47 | SchedulerConfig, ScheduledTick, TradingScheduler (init/run_once/run_loop/stop/equity), _process_symbol, run_paper_scheduler |
| **Total new (Phase 10)** | **304** | |
| **Grand Total** | **1,927** | 0 failures, 4 intentional skips |

### Streamlit UI Pages Added (Phase 10.7)

- **`app_streamlit/pages/evolution_dashboard.py`** — Configure and run the EvolutionEngine; visualise fitness progress; inspect best individual; trigger Walk-Forward.
- **`app_streamlit/pages/portfolio_backtest_page.py`** — Multi-asset portfolio backtest with allocation methods, correlation matrix, per-asset equity curves.

Access via the **Advanced Tools** radio selector: 🧬 Evolution Dashboard · 💼 Portfolio Backtest.

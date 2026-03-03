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

# Phi-nance Strategy Implementation Prompt

## Your Role

You are a quantitative trading strategy developer. Your job is to implement new
prediction strategies for the **Phi-nance** backtesting framework. Each strategy
must follow the exact patterns and conventions already established in the codebase.

---

## Project Context

**Phi-nance** is a backtesting framework that evaluates trading strategies by
**prediction accuracy** — how well each strategy predicts next-day price direction
— rather than P&L. It uses:

- **Lumibot** for backtesting with Yahoo Finance data (free, daily OHLCV bars)
- **Streamlit** for an interactive dashboard
- **PredictionMixin** to track UP/DOWN/NEUTRAL predictions and score them

The strategies you implement are inspired by the **FINAL_GNOSIS** trading system
(github.com/DGator86/FINAL_GNOSIS), which contains institutional-grade
implementations of these methodologies. Your job is to distill each methodology
into a clean, self-contained Lumibot strategy that works with daily OHLCV data
from Yahoo Finance.

---

## Architecture You Must Follow

### File Structure

```
strategies/
├── prediction_tracker.py    # DO NOT MODIFY — PredictionMixin + scoring
├── buy_and_hold.py          # Existing baseline
├── momentum.py              # Existing
├── mean_reversion.py        # Existing
├── rsi.py                   # Existing
├── bollinger.py             # Existing
├── macd.py                  # Existing
├── dual_sma.py              # Existing
├── breakout.py              # Existing
├── wyckoff.py               # Existing
├── liquidity_pools.py       # Existing
├── <your_new_strategy>.py   # NEW — one file per strategy
```

### Strategy Template (follow exactly)

Every strategy must follow this pattern:

```python
"""
Strategy Name
--------------
One-paragraph description of what this strategy does and the core idea
behind it.

Usage:
    python strategies/<filename>.py
"""

from datetime import datetime

from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy

from strategies.prediction_tracker import PredictionMixin


class YourStrategy(PredictionMixin, Strategy):
    parameters = {
        "symbol": "SPY",
        # strategy-specific params with sensible defaults
    }

    def initialize(self):
        self.sleeptime = "1D"
        self._init_predictions()

    def on_trading_iteration(self):
        symbol = self.parameters["symbol"]
        # ... fetch bars, compute indicators ...

        bars = self.get_bars([symbol], NEEDED_BARS, timestep="day")
        if not bars:
            return

        asset_key = next((a for a in bars if a.symbol == symbol), None)
        if asset_key is None:
            return

        df = bars[asset_key].df
        if len(df) < MINIMUM_REQUIRED:
            return

        current_price = self.get_last_price(symbol)
        position = self.get_position(symbol)

        # CORE LOGIC: compute your signal
        # Must call ONE of these every iteration:
        #   self.record_prediction(symbol, "UP", current_price)
        #   self.record_prediction(symbol, "DOWN", current_price)
        #   self.record_prediction(symbol, "NEUTRAL", current_price)

        # TRADING: buy on UP, sell on DOWN
        if signal == "UP":
            self.record_prediction(symbol, "UP", current_price)
            if position is None:
                quantity = int(self.portfolio_value * 0.95 // current_price)
                if quantity > 0:
                    order = self.create_order(symbol, quantity, "buy")
                    self.submit_order(order)
        elif signal == "DOWN":
            self.record_prediction(symbol, "DOWN", current_price)
            if position is not None:
                self.sell_all()
        else:
            self.record_prediction(symbol, "NEUTRAL", current_price)

    def trace_stats(self, context, snapshot_before):
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
        }


if __name__ == "__main__":
    backtesting_start = datetime(2020, 1, 1)
    backtesting_end = datetime(2024, 12, 31)

    YourStrategy.backtest(
        YahooDataBacktesting,
        backtesting_start,
        backtesting_end,
        benchmark_asset="SPY",
    )
```

### Integration Checklist (for each new strategy)

After creating the strategy file, you must also:

1. **`run_backtest.py`** — Add import and entry to the `STRATEGIES` dict
2. **`dashboard.py`** — Add import and entry to the `STRATEGY_CATALOG` dict with
   description and parameter specs (follow the exact format of existing entries)
3. **`dashboard.py` `resolve_params()`** — Add any integer parameter keys to the
   `int_keys` tuple if they aren't already listed

---

## Strategies to Implement

Below are the strategies to build, ordered from simpler to more complex. Each
includes the methodology, the signal logic, and the parameters to expose.

---

### Strategy 1: Volume Spread Analysis (VSA)

**File:** `strategies/vsa.py`
**Class:** `VSAStrategy`

**Core Idea (Wyckoff-derived):**
Analyze the relationship between price range (high - low) and volume to detect
smart money activity. When range and volume diverge, it signals potential
reversals.

**Signal Logic:**
1. Compute the Average True Range (ATR) over `atr_period` bars
2. Compute average volume over `vol_period` bars
3. Classify today's candle:
   - **Wide Range**: today's range > `range_mult` * ATR
   - **Narrow Range**: today's range < ATR / `range_mult`
   - **High Volume**: today's volume > `vol_mult` * avg volume
   - **Low Volume**: today's volume < avg volume / `vol_mult`
4. Generate signals:
   - **Climax Selling** (UP): Wide range + High volume + bearish close
     (close < open) + price in bottom 30% of lookback range = smart money
     absorbing selling -> predict UP
   - **Climax Buying** (DOWN): Wide range + High volume + bullish close
     (close > open) + price in top 70% of lookback range = smart money
     distributing -> predict DOWN
   - **No Supply** (UP): Narrow range + Low volume + price near support
     (bottom 40% of range) = sellers exhausted -> predict UP
   - **No Demand** (DOWN): Narrow range + Low volume + price near resistance
     (top 60% of range) = buyers exhausted -> predict DOWN
   - Otherwise: NEUTRAL

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 30,       # bars for range context
    "atr_period": 14,     # ATR calculation period
    "vol_period": 20,     # average volume period
    "range_mult": 1.5,    # multiplier for wide/narrow classification
    "vol_mult": 1.5,      # multiplier for high/low volume classification
}
```

**Dashboard entry:**
```python
"Volume Spread Analysis": {
    "class": VSAStrategy,
    "description": (
        "Analyzes range-vs-volume divergence to detect smart money. "
        "Climax reversals and no-supply/no-demand signals."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 30, "min": 10, "max": 120},
        "atr_period": {"label": "ATR Period", "type": "number", "default": 14, "min": 5, "max": 50},
        "vol_period": {"label": "Volume Avg Period", "type": "number", "default": 20, "min": 5, "max": 100},
    },
}
```

---

### Strategy 2: Fair Value Gap (FVG)

**File:** `strategies/fvg.py`
**Class:** `FairValueGapStrategy`

**Core Idea (ICT concept):**
A Fair Value Gap is a three-candle pattern where the middle candle moves so
aggressively that a gap is left between candle 1 and candle 3 (their shadows
don't overlap). Price tends to return and fill these gaps.

**Signal Logic:**
1. Scan the last `lookback` bars for FVG patterns:
   - **Bullish FVG (BISI)**: `bar[i-2].high < bar[i].low` (gap up through
     middle candle). This gap acts as support.
   - **Bearish FVG (SIBI)**: `bar[i-2].low > bar[i].high` (gap down through
     middle candle). This gap acts as resistance.
2. Track open (unfilled) gaps. A gap is "filled" when price returns through it.
3. Signal generation:
   - If current price is sitting inside (or just touched) a bullish FVG that
     hasn't been filled yet -> predict **UP** (expecting a bounce from support)
   - If current price is sitting inside (or just touched) a bearish FVG that
     hasn't been filled yet -> predict **DOWN** (expecting rejection from
     resistance)
   - Multiple unfilled gaps: use the nearest one to current price
   - No active gaps near price: **NEUTRAL**
4. A gap is considered "near" if price is within `proximity_pct`% of the gap.

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 50,          # bars to scan for FVGs
    "min_gap_pct": 0.001,    # minimum gap size as % of price
    "proximity_pct": 0.005,  # how close price must be to a gap to trigger
    "max_gaps": 10,          # max unfilled gaps to track
}
```

**Dashboard entry:**
```python
"Fair Value Gap (ICT)": {
    "class": FairValueGapStrategy,
    "description": (
        "Detects 3-candle Fair Value Gaps (ICT concept). "
        "Predicts UP at bullish gaps, DOWN at bearish gaps."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 50, "min": 20, "max": 200},
        "min_gap_pct": {"label": "Min Gap Size (%)", "type": "number", "default": 0.1, "min": 0.01, "max": 1.0},
        "proximity_pct": {"label": "Proximity (%)", "type": "number", "default": 0.5, "min": 0.1, "max": 2.0},
    },
}
```

Note: `min_gap_pct` and `proximity_pct` in the dashboard are displayed as
percentages (0.1 = 0.1%) but stored internally as decimals (0.001). Convert
in `resolve_params` or in the strategy itself.

---

### Strategy 3: Order Block

**File:** `strategies/order_block.py`
**Class:** `OrderBlockStrategy`

**Core Idea (ICT concept):**
An Order Block is the last opposing candle before a strong impulsive move. These
represent zones where institutional orders were placed and price tends to return
to them.

**Signal Logic:**
1. Identify **impulsive moves**: a sequence of `impulse_bars` or more
   consecutive same-direction candles with above-average range.
2. The **order block** is the last candle that went the opposite direction
   before the impulse:
   - **Bullish OB**: Last bearish candle before a strong bullish impulse.
     The OB zone is [low, high] of that candle.
   - **Bearish OB**: Last bullish candle before a strong bearish impulse.
     The OB zone is [low, high] of that candle.
3. Track active (unmitigated) order blocks. An OB is "mitigated" when price
   fully passes through its zone.
4. Signal:
   - Price returns to a bullish OB zone (price between OB low and OB high)
     -> predict **UP**
   - Price returns to a bearish OB zone -> predict **DOWN**
   - No active OB near price -> **NEUTRAL**

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 50,       # bars to scan
    "impulse_bars": 3,    # min consecutive same-direction bars for impulse
    "max_blocks": 10,     # max active OBs to track
}
```

**Dashboard entry:**
```python
"Order Block (ICT)": {
    "class": OrderBlockStrategy,
    "description": (
        "Finds institutional Order Blocks (last opposing candle before "
        "an impulse). Predicts bounce when price returns to the block."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 50, "min": 20, "max": 200},
        "impulse_bars": {"label": "Impulse Bars", "type": "number", "default": 3, "min": 2, "max": 6},
    },
}
```

---

### Strategy 4: Supply & Demand Zones

**File:** `strategies/supply_demand.py`
**Class:** `SupplyDemandStrategy`

**Core Idea:**
Identify zones where significant supply/demand shifts occurred based on price
structure. Demand zones form at the low between two highs where the second high
exceeds the first (HH validation). Supply zones form at the high between two
lows where the second low undercuts the first (LL validation).

**Signal Logic:**
1. Find swing highs and swing lows using `swing_strength` bars on each side
2. Identify **demand zones**: For each swing low, check if there's a swing high
   before AND after it, and the later high > earlier high. The zone is the area
   around that swing low: [swing_low - buffer, swing_low + buffer] where
   buffer = ATR * 0.5
3. Identify **supply zones**: For each swing high, check if there's a swing low
   before AND after it, and the later low < earlier low. Zone around swing high.
4. Track zone freshness: FRESH (never tested) > TESTED (price returned once) >
   BROKEN (price passed through)
5. Signal:
   - Price enters a FRESH or TESTED demand zone -> predict **UP**
   - Price enters a FRESH or TESTED supply zone -> predict **DOWN**
   - No active zone near price -> **NEUTRAL**

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 60,
    "swing_strength": 3,
    "max_zones": 10,
}
```

**Dashboard entry:**
```python
"Supply & Demand Zones": {
    "class": SupplyDemandStrategy,
    "description": (
        "Finds validated supply/demand zones (HH/LL structure). "
        "Predicts UP at demand zones, DOWN at supply zones."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 60, "min": 20, "max": 200},
        "swing_strength": {"label": "Swing Strength", "type": "number", "default": 3, "min": 2, "max": 8},
    },
}
```

---

### Strategy 5: Cumulative Volume Delta (CVD) Divergence

**File:** `strategies/cvd_divergence.py`
**Class:** `CVDDivergenceStrategy`

**Core Idea (Order Flow concept):**
Track cumulative buying vs selling pressure using volume and price action as a
proxy (since we only have OHLCV data). When price makes new highs but CVD fails
to confirm, it signals exhaustion.

**Signal Logic:**
1. Estimate delta for each bar:
   - If close > open (bullish): delta = volume * ((close - low) / (high - low))
     minus volume * ((high - close) / (high - low))
   - If close < open (bearish): delta is negative by same logic
   - Handle edge case where high == low (delta = 0)
2. Compute cumulative delta (running sum) over `lookback` bars
3. Smooth CVD with a `smoothing` period SMA
4. Detect divergence:
   - **Bearish divergence** (DOWN): Price made a higher high over last
     `divergence_window` bars, but CVD made a lower high
   - **Bullish divergence** (UP): Price made a lower low over last
     `divergence_window` bars, but CVD made a higher low
5. Detect exhaustion:
   - CVD slope flattening (change < 10% of average change) while price still
     trending -> exhaustion signal in opposite direction
6. No divergence or exhaustion -> **NEUTRAL**

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 40,
    "smoothing": 14,
    "divergence_window": 10,
}
```

**Dashboard entry:**
```python
"CVD Divergence": {
    "class": CVDDivergenceStrategy,
    "description": (
        "Tracks cumulative volume delta for price/volume divergence. "
        "Predicts reversals when price and buying pressure disagree."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 40, "min": 20, "max": 120},
        "smoothing": {"label": "CVD Smoothing", "type": "number", "default": 14, "min": 5, "max": 30},
        "divergence_window": {"label": "Divergence Window", "type": "number", "default": 10, "min": 5, "max": 30},
    },
}
```

---

### Strategy 6: Liquidity Sweep & Reversal

**File:** `strategies/liquidity_sweep.py`
**Class:** `LiquiditySweepStrategy`

**Core Idea (ICT / Liquidity Concepts):**
Extends the existing `liquidity_pools.py` with more sophisticated sweep
detection, inducement classification, and reversal confirmation.

**Signal Logic:**
1. Find swing highs/lows as liquidity pools (same as liquidity_pools.py)
2. Classify pools by strength:
   - **Clustered** (multiple swings within 0.3% of each other): strength = HIGH
   - **Equal highs/lows** (2+ swings at nearly identical price): strength = HIGH
   - **Single swing**: strength = NORMAL
3. Detect sweeps with **reversal confirmation**:
   - **Buy-side sweep**: today's high pierced a swing high BUT close is back
     below it, AND the close is in the lower half of today's range (strong
     rejection). This is a bearish reversal signal -> predict **DOWN**
   - **Sell-side sweep**: today's low pierced a swing low BUT close is back
     above it, AND the close is in the upper half of today's range (strong
     rejection). This is a bullish reversal signal -> predict **UP**
4. Classify inducement pattern type:
   - If sweep of a HIGH-strength pool + strong rejection: **Stop Hunt** (highest
     confidence)
   - If sweep but weak rejection (close near middle of range): **False Breakout**
     (lower confidence) — still signal but weaker
5. If no sweep: check if price is being "drawn" toward nearest pool (same logic
   as existing liquidity_pools.py)
6. Priority: Sweep reversals override draw-toward signals.

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 50,
    "swing_strength": 3,
    "cluster_pct": 0.003,     # 0.3% threshold for clustering
    "require_rejection": True, # require close in opposite half of range
}
```

**Dashboard entry:**
```python
"Liquidity Sweep": {
    "class": LiquiditySweepStrategy,
    "description": (
        "Detects liquidity sweeps with reversal confirmation. "
        "Identifies stop hunts at clustered swing levels."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 50, "min": 20, "max": 150},
        "swing_strength": {"label": "Swing Strength", "type": "number", "default": 3, "min": 2, "max": 8},
    },
}
```

---

### Strategy 7: Premium/Discount + OTE

**File:** `strategies/premium_discount.py`
**Class:** `PremiumDiscountStrategy`

**Core Idea (ICT concept):**
Divide the recent price range into premium (top half) and discount (bottom half)
zones. Only look for longs in the discount zone and shorts in the premium zone.
The Optimal Trade Entry (OTE) is the sweet spot at the 0.62-0.79 Fibonacci
retracement level.

**Signal Logic:**
1. Find the highest high and lowest low over `lookback` bars
2. Calculate equilibrium: (highest + lowest) / 2
3. Calculate OTE zone:
   - For bullish (after a swing low): OTE between 0.62 and 0.79 retracement
     of the recent swing
   - For bearish (after a swing high): same Fibonacci levels from the top
4. Signal:
   - Price is in discount zone (below equilibrium) AND within OTE retracement
     zone AND there's a bullish candle (close > open) -> predict **UP**
   - Price is in premium zone (above equilibrium) AND within OTE retracement
     zone AND there's a bearish candle (close < open) -> predict **DOWN**
   - Price in discount but not at OTE -> weak UP (still predict UP)
   - Price in premium but not at OTE -> weak DOWN (still predict DOWN)
   - Price at equilibrium (within 2% band) -> **NEUTRAL**

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 40,
    "ote_high": 0.62,     # OTE zone start (Fibonacci)
    "ote_low": 0.79,      # OTE zone end (Fibonacci)
    "eq_band": 0.02,      # equilibrium neutral band (2%)
}
```

**Dashboard entry:**
```python
"Premium/Discount + OTE": {
    "class": PremiumDiscountStrategy,
    "description": (
        "ICT Premium/Discount zones with Optimal Trade Entry. "
        "Longs in discount, shorts in premium, best at OTE fib levels."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 40, "min": 15, "max": 120},
    },
}
```

---

### Strategy 8: PENTA Confluence

**File:** `strategies/penta_confluence.py`
**Class:** `PentaConfluenceStrategy`

**Core Idea (FINAL_GNOSIS signature methodology):**
Combine signals from 5 sub-strategies (Wyckoff, ICT-style, Order Flow proxy,
Supply/Demand, Liquidity) and boost confidence when multiple agree. This is the
crown jewel strategy — it synthesizes all methodologies.

**Signal Logic:**
1. On each iteration, compute 5 independent sub-signals. Each returns UP, DOWN,
   or NEUTRAL:
   - **Wyckoff signal**: Same logic as existing `wyckoff.py` (position in range
     + volume on up vs down days)
   - **FVG signal**: Simplified — check for 3-bar gap pattern in last 20 bars,
     is price near an unfilled gap?
   - **Supply/Demand signal**: Is price near a validated swing zone?
   - **Liquidity signal**: Same logic as existing `liquidity_pools.py` (nearest
     pool + sweep detection)
   - **Momentum signal**: Simple — is price above or below its 20-day SMA?
2. Count agreement:
   - Count how many say UP, how many say DOWN, how many say NEUTRAL
3. Apply confluence thresholds:
   - **PENTA (5/5 agree)**: Strong signal (all 5 agree on direction)
   - **QUAD (4/5)**: Strong signal
   - **TRIPLE (3/5)**: Moderate signal
   - **DOUBLE (2/5)**: Weak — only signal if the other 3 are NEUTRAL (not
     opposing)
   - **Less than 2 agree or conflicting**: NEUTRAL
4. Final prediction:
   - If UP count >= 3: predict **UP**
   - If DOWN count >= 3: predict **DOWN**
   - If UP count == 2 and DOWN count == 0: predict **UP**
   - If DOWN count == 2 and UP count == 0: predict **DOWN**
   - Otherwise: **NEUTRAL**

**Parameters:**
```python
parameters = {
    "symbol": "SPY",
    "lookback": 40,
    "swing_strength": 3,
    "sma_period": 20,
    "range_pct": 0.3,  # for Wyckoff sub-signal
}
```

**Dashboard entry:**
```python
"PENTA Confluence": {
    "class": PentaConfluenceStrategy,
    "description": (
        "Combines 5 methodologies (Wyckoff, FVG, Supply/Demand, "
        "Liquidity, Momentum). Signals when 3+ agree on direction."
    ),
    "params": {
        "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
        "lookback": {"label": "Lookback (days)", "type": "number", "default": 40, "min": 20, "max": 120},
        "swing_strength": {"label": "Swing Strength", "type": "number", "default": 3, "min": 2, "max": 8},
        "sma_period": {"label": "SMA Period", "type": "number", "default": 20, "min": 5, "max": 100},
    },
}
```

---

## Important Constraints

1. **Data source**: You only have daily OHLCV bars from Yahoo Finance. No
   tick data, no bid/ask, no Level 2, no options chain data. Adapt all
   methodologies to work with this data.

2. **No external dependencies**: Only use pandas, numpy, and what Lumibot
   provides. Do not add new packages to requirements.txt.

3. **One prediction per iteration**: Each strategy must call
   `self.record_prediction()` exactly once per `on_trading_iteration()` call.

4. **Keep it simple**: These are backtesting strategies, not production trading
   systems. Favor clarity over sophistication. Each file should be 80-150 lines.

5. **Position sizing**: Use the standard pattern:
   `quantity = int(self.portfolio_value * 0.95 // current_price)`

6. **Don't modify existing files** (other than the required integration into
   `run_backtest.py` and `dashboard.py`). Don't refactor the existing
   strategies. Only add new files and the minimal integration edits.

7. **Parameter naming**: Use snake_case. Add new integer param keys to the
   `int_keys` tuple in `dashboard.py:resolve_params()`.

8. **Testing**: After implementing each strategy, verify it can run standalone:
   `python strategies/<filename>.py` (uses the `if __name__ == "__main__"` block).

---

## Implementation Order

Implement in this order (each builds understanding for the next):

1. `vsa.py` — Volume Spread Analysis (pure candle + volume analysis)
2. `fvg.py` — Fair Value Gap (3-candle pattern recognition)
3. `order_block.py` — Order Block (impulse + preceding candle detection)
4. `supply_demand.py` — Supply & Demand Zones (swing structure validation)
5. `cvd_divergence.py` — CVD Divergence (volume delta estimation from OHLCV)
6. `liquidity_sweep.py` — Liquidity Sweep (enhanced version of liquidity_pools)
7. `premium_discount.py` — Premium/Discount + OTE (Fibonacci zone trading)
8. `penta_confluence.py` — PENTA Confluence (meta-strategy combining all)

After all 8 strategies are implemented, update `run_backtest.py` and
`dashboard.py` with all new entries.

---

## Reference: How Existing Strategies Are Registered

**In `run_backtest.py`:**
```python
from strategies.wyckoff import WyckoffStrategy

STRATEGIES = {
    ...
    "wyckoff": WyckoffStrategy,
}
```

**In `dashboard.py`:**
```python
from strategies.wyckoff import WyckoffStrategy

STRATEGY_CATALOG = {
    ...
    "Wyckoff": {
        "class": WyckoffStrategy,
        "description": "...",
        "params": {
            "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
            "lookback": {"label": "Lookback (days)", "type": "number", "default": 30, "min": 10, "max": 120},
        },
    },
}
```

**In `dashboard.py` `resolve_params()`:**
```python
int_keys = (
    "lookback_days", "rebalance_days", "sma_period", "rsi_period",
    "oversold", "overbought", "bb_period", "fast_period", "slow_period",
    "signal_period", "channel_period", "lookback", "swing_strength",
)
```

Add your new integer param names (like `atr_period`, `vol_period`, `impulse_bars`,
`smoothing`, `divergence_window`, `max_zones`, `max_blocks`, `max_gaps`) to this
tuple.

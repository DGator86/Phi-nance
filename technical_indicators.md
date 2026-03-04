# Mainstream Technical Indicators

## Trend Indicators

| Indicator | What it does | How it works |
|---|---|---|
| **SMA** (Simple Moving Average) | Smooths price to show trend direction | Arithmetic mean of closing prices over N periods |
| **EMA** (Exponential Moving Average) | Like SMA but reacts faster to recent prices | Applies a multiplier `2/(N+1)` to weight recent prices more heavily |
| **WMA** (Weighted Moving Average) | Like EMA but linearly weighted | Most recent bar gets weight N, second-most-recent N-1, etc. |
| **DEMA** (Double EMA) | Reduces lag of EMA | `2*EMA(price) - EMA(EMA(price))` |
| **TEMA** (Triple EMA) | Further reduces lag | `3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))` |
| **VWMA** (Volume-Weighted MA) | Weights average by volume | `sum(price * volume) / sum(volume)` over N periods |
| **Hull MA** | Very low-lag moving average | `WMA(2*WMA(n/2) - WMA(n), sqrt(n))` |
| **Ichimoku Cloud** | Multi-component trend/support/resistance system | Five lines: Tenkan (9-period midpoint), Kijun (26-period midpoint), Senkou A/B (future projections), Chikou (lagging close). Cloud (kumo) shows S/R zones |
| **Parabolic SAR** | Trailing stop that flips sides on trend reversal | Dot follows price with an acceleration factor; flips above/below price when trend reverses |
| **ADX** (Average Directional Index) | Measures trend *strength*, not direction | Derived from +DI and -DI (directional movement); ADX > 25 = strong trend |
| **Supertrend** | Trend-following stop/reversal line | Based on ATR bands above/below price; line flips when price crosses it |
| **Aroon** | Detects trend onset and strength | Aroon Up = bars since 25-period high / 25; Aroon Down = bars since 25-period low / 25 |

---

## Momentum / Oscillators

| Indicator | What it does | How it works |
|---|---|---|
| **RSI** (Relative Strength Index) | Measures speed and magnitude of price changes; overbought/oversold | `100 - 100/(1 + avg_gain/avg_loss)` over 14 periods; >70 = overbought, <30 = oversold |
| **MACD** (Moving Avg Convergence Divergence) | Shows momentum via EMA crossovers | `MACD line = EMA(12) - EMA(26)`; Signal = `EMA(9)` of MACD; Histogram = MACD - Signal |
| **Stochastic Oscillator** | Compares close to recent high-low range | `%K = (close - lowest_low) / (highest_high - lowest_low) * 100`; `%D = SMA(3)` of %K |
| **Stochastic RSI** | RSI of RSI — more sensitive | Applies the Stochastic formula to RSI values instead of price |
| **CCI** (Commodity Channel Index) | Measures deviation from average price | `(typical_price - SMA) / (0.015 * mean_deviation)`; ±100 are key thresholds |
| **Williams %R** | Inverse Stochastic; overbought/oversold | `(highest_high - close) / (highest_high - lowest_low) * -100`; 0 to -100 scale |
| **ROC** (Rate of Change) | Raw momentum — % change over N periods | `(close - close[N]) / close[N] * 100` |
| **Momentum** | Absolute price change over N periods | `close - close[N]` |
| **TSI** (True Strength Index) | Double-smoothed momentum | Double EMA of price change divided by double EMA of absolute price change; ranges ±100 |
| **Ultimate Oscillator** | Combines short/medium/long momentum | Weighted sum of buying pressure vs. true range across 7, 14, and 28 periods |
| **PPO** (Percentage Price Oscillator) | MACD expressed as a percentage | `(EMA_fast - EMA_slow) / EMA_slow * 100` |
| **DPO** (Detrended Price Oscillator) | Removes trend to isolate cycles | `close - SMA(N/2 + 1 bars ago)`; shows cycles, not trend |

---

## Volatility Indicators

| Indicator | What it does | How it works |
|---|---|---|
| **Bollinger Bands** | Envelope around price showing relative high/low volatility | Upper/Lower = `SMA(20) ± 2 * stddev(20)`; bands expand in volatile markets, contract in quiet ones |
| **ATR** (Average True Range) | Measures market volatility | True Range = `max(high-low, |high-prev_close|, |low-prev_close|)`; ATR = smoothed average of TR |
| **Keltner Channels** | Volatility bands using ATR instead of stddev | `EMA(20) ± 2*ATR(10)`; when price exits channel, signals breakout |
| **Donchian Channels** | Tracks highest high / lowest low over N periods | Upper = highest high(N), Lower = lowest low(N), Middle = average; classic turtle-trading basis |
| **Historical Volatility (HV)** | Annualized stddev of returns | Stddev of log returns over N periods, scaled by `sqrt(252)` |
| **Chaikin Volatility** | Rate of change of the high-low spread | EMA of (high-low), then ROC of that EMA |

---

## Volume Indicators

| Indicator | What it does | How it works |
|---|---|---|
| **OBV** (On-Balance Volume) | Cumulative volume pressure | Adds volume on up-days, subtracts on down-days; divergence from price signals reversals |
| **VWAP** (Volume-Weighted Average Price) | Intraday fair-value benchmark | `sum(typical_price * volume) / sum(volume)` since session open; resets each day |
| **Chaikin Money Flow (CMF)** | Measures buying/selling pressure | Money Flow Volume = `((close-low)-(high-close))/(high-low) * volume`; CMF = sum over N / sum of volume |
| **MFI** (Money Flow Index) | Volume-weighted RSI | Like RSI but uses typical price * volume instead of just price |
| **Accumulation/Distribution (A/D)** | Tracks cumulative money flow | `Close Location Value = ((close-low)-(high-close))/(high-low)`; A/D += CLV * volume |
| **Volume Oscillator** | Trend in volume | `(fast_vol_MA - slow_vol_MA) / slow_vol_MA * 100` |
| **VPVR / Volume Profile** | Shows price levels with most traded volume | Horizontal histogram of volume at each price level; high-volume nodes = S/R |
| **Force Index** | Combines price change and volume | `(close - prev_close) * volume`; smoothed with EMA |
| **Ease of Movement (EoM)** | Measures how easily price moves | `midpoint_move / volume_ratio`; high value = price moving easily on light volume |

---

## Support / Resistance

| Indicator | What it does | How it works |
|---|---|---|
| **Pivot Points** (Classic) | Calculates S/R levels from prior session | `P = (H+L+C)/3`; R1 = `2P-L`, S1 = `2P-H`; R2, S2 extend further |
| **Fibonacci Retracement** | Key S/R levels based on Fibonacci ratios | Draws horizontal lines at 23.6%, 38.2%, 50%, 61.8%, 78.6% of a price swing |
| **Fibonacci Extensions** | Projects price targets beyond a swing | 127.2%, 161.8%, 261.8% extensions of the measured swing |
| **VWAP Anchored** | VWAP from a user-chosen anchor point | Same as VWAP but starts from a specific bar (e.g., earnings date, swing low) |
| **Camarilla Pivots** | Tighter intraday S/R levels | Uses `(H-L) * factor + C`; factors derived from 1/1.1 ratios |

---

## Breadth / Market Structure

| Indicator | What it does | How it works |
|---|---|---|
| **A/D Line** (Advance-Decline) | Market breadth — how many stocks are participating | Cumulative sum of `(advancing issues - declining issues)` each day |
| **McClellan Oscillator** | Short-term breadth momentum | `EMA(19) - EMA(39)` of daily advances minus declines |
| **New Highs - New Lows** | Strength of trend across market | Count of stocks making 52-week highs minus 52-week lows |
| **TRIN (Arms Index)** | Intraday sentiment indicator | `(advancing/declining) / (advancing_vol/declining_vol)`; >1 = bearish, <1 = bullish |

---

## Miscellaneous / Composite

| Indicator | What it does | How it works |
|---|---|---|
| **Heikin-Ashi** | Smoothed candlestick representation | `HA_close = (O+H+L+C)/4`; `HA_open = (prev_HA_open + prev_HA_close)/2`; reduces noise |
| **Renko** | Noise-filtered brick chart | Plots a new brick only when price moves a fixed amount (brick size); ignores time |
| **Kagi** | Trend-reversal chart | Line reverses direction only when price reverses by a threshold amount |
| **Elder Ray Index** | Bull/bear power around EMA | Bull Power = `high - EMA(13)`; Bear Power = `low - EMA(13)` |
| **Mass Index** | Detects trend reversals via range widening | Sums ratio of `EMA(9, high-low) / EMA(9, EMA(9, high-low))` over 25 periods; "reversal bulge" when sum > 27 then drops below 26.5 |
| **Coppock Curve** | Long-term buy signals for indices | Weighted moving average of the sum of 14-month and 11-month ROC; buy when it turns up from below zero |
| **PSAR + ADX combo** | Trend filter + reversal signal | Common pairing: only take Parabolic SAR signals when ADX confirms strong trend |
| **Chande Momentum Oscillator (CMO)** | Raw momentum without smoothing | `(sum_up - sum_down) / (sum_up + sum_down) * 100` over N periods; ranges -100 to +100 |
| **Klinger Oscillator** | Long-term money flow + short-term reversals | Multiplies volume by trend direction (+1/-1) based on typical price, then takes EMA difference |

---

## Key Groupings

- **Trend-following** (MA, MACD, Ichimoku, ADX) — work well in trending markets, give false signals in ranges
- **Mean-reversion oscillators** (RSI, Stochastic, CCI, Williams %R) — work well in ranging markets, get crushed in strong trends
- **Volatility** (ATR, Bollinger, Keltner) — used to size positions and identify breakout conditions
- **Volume** (OBV, VWAP, MFI) — confirm or deny price moves; divergence is a classic warning signal

# Phi-nance — Live Backtest Workbench

> Premium quant SaaS-grade backtesting platform.
> Dark mode · Purple/Orange · Regime-aware · PhiAI auto-tuning · Run history.

---

## Quick Start

```bash
# Install dependencies
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch the Live Backtest Workbench
streamlit run app_streamlit/live_workbench.py

# Launch the legacy dashboard (preserved)
streamlit run dashboard.py
```

---

## Live Backtest Workbench

The workbench lives at `app_streamlit/live_workbench.py`.

### Workflow (5 steps)

| Step | Panel | What happens |
|---|---|---|
| 1 | **Dataset Builder** | Fetch OHLCV from yfinance/Alpha Vantage. Cache as parquet. Never re-fetch. |
| 2 | **Indicator Selection** | Choose from 14+ indicators. Set params manually or enable PhiAI auto-tune. |
| 3 | **Blending Panel** | Weighted sum / voting / regime-weighted / PhiAI blend mode. |
| 4 | **PhiAI Panel** | Full-auto: indicator selection + parameter optimization + blend weights. |
| 5 | **Backtest Controls** | Equities or options mode. SL/TP/trailing stop. Run live with progress bar. |

### Results

- **Summary tab** — start/end capital, net P&L $/%,  primary metric highlighted
- **Equity Curve** — portfolio value + drawdown chart
- **Trades** — per-trade log with entry/exit/P&L + bar charts
- **Metrics** — full metric table (Sharpe, Sortino, Calmar, Profit Factor, Win Rate, Direction Accuracy...)
- **Diagnostics** — individual signals, blended signal, position series, PhiAI explanation

### Export

- `config.json` — full RunConfig for reproducibility
- `trades.csv` — trade-level data

---

## Architecture

See [Architecture.md](Architecture.md) for full module breakdown.

```
phi/
  data/        — Dataset fetch + cache
  indicators/  — 14+ indicators → normalized [-1,+1] signals
  blending/    — Signal blending engine
  backtest/    — Vectorized engine + RunConfig + RunHistory
  options/     — Black-Scholes options simulator
  phiai/       — Random-search auto-tuner

app_streamlit/
  live_workbench.py  — Main workbench UI
  styles.py          — Dark theme CSS

data_cache/    — Cached parquet files + metadata JSON
runs/          — Reproducible run storage (config + results + trades)
```

---

## Indicators Available

| Name | Display | Type |
|---|---|---|
| `rsi` | RSI | Bounded Oscillator |
| `macd` | MACD | Momentum |
| `bollinger` | Bollinger Bands | Bounded Oscillator |
| `stochastic` | Stochastic | Bounded Oscillator |
| `dual_sma` | Dual SMA Crossover | Momentum |
| `ema_crossover` | EMA Crossover | Momentum |
| `momentum` | Momentum | Momentum |
| `roc` | Rate of Change | Momentum |
| `atr_ratio` | ATR Ratio | Momentum |
| `vwap_dev` | VWAP Deviation | Price Level |
| `cmf` | Chaikin Money Flow | Bounded Oscillator |
| `adx` | ADX / DI | Momentum |
| `wyckoff` | Wyckoff | Discrete State |
| `range_pos` | Range Position | Bounded Oscillator |
| `phi_mft` | Phi-Bot (MFT) | Full MFT composite |

---

## Blend Modes

| Mode | Description |
|---|---|
| `weighted_sum` | Linear combination with user-defined weights |
| `voting` | Weighted majority vote of signal directions |
| `regime_weighted` | Per-regime weight matrix × MFT regime probabilities |
| `phiai` | PhiAI assigns weights based on individual indicator scores |

---

## PhiAI

PhiAI (`phi/phiai/tuner.py`) uses random search to:
1. Score each indicator individually on the chosen metric
2. Select the best N indicators (respecting drawdown cap)
3. Tune each indicator's parameters via random search
4. Output blend weights proportional to scores

Constraints: max indicators, drawdown cap, no-short flag.

---

## Data Storage

```
data_cache/{vendor}/{symbol}/{timeframe}/{YYYYMMDD}_{YYYYMMDD}.parquet
data_cache/{vendor}/{symbol}/{timeframe}/{YYYYMMDD}_{YYYYMMDD}_metadata.json

runs/{run_id}/
  config.json    — full RunConfig
  results.json   — metrics + equity curve
  trades.csv     — per-trade log
```

---

## Theme

All UI enforces dark mode:
- Background: `#0a0a12` (near-black)
- Primary: `#a855f7` (purple) + glow effects on active elements
- Accent: `#f97316` (orange) for highlights
- Cards with `#16162a` + subtle `#2d2d50` borders

---

## Legacy (preserved)

The original Lumibot-backed strategies and dashboard are fully preserved:

```bash
python run_backtest.py --strategy rsi --start 2022-01-01 --end 2024-01-01 --budget 50000
streamlit run dashboard.py
```

Strategies: `buy_and_hold`, `momentum`, `mean_reversion`, `rsi`, `bollinger`, `macd`, `dual_sma`, `breakout`, `wyckoff`, `liquidity_pools`

---

## VPS Deployment

See [STEP_BY_STEP.md](STEP_BY_STEP.md) for VPS setup.

The workbench runs on port `8501` (same as before):

```bash
screen -S workbench
source venv/bin/activate
streamlit run app_streamlit/live_workbench.py
# Ctrl+A, D to detach
```

# Step 7: Full Options Integration Guide

This guide is the next layer after getting the PostgreSQL options vendor and a basic strategy working. It focuses on three integration tracks inside the current Phi-nance codebase:

1. Extending backtesting flows for richer options lifecycle handling.
2. Expanding the Streamlit workbench for options exploration.
3. Building signal pipelines from Greeks, implied volatility, and open interest.

## 7.1 Extend the backtesting stack for options

Phi-nance already has multiple options-related execution paths (`phi/backtest/options_engine.py`, `phi/options/backtest.py`, and `phi/options/engine_backtest.py`). Use these as your starting points instead of creating an isolated one-off backtester.

### 1) Map the current trade representation

Review the data model and execution loop in:

- `phi/backtest/options_engine.py`
- `phi/backtest/portfolio.py`
- `phi/options/position.py`
- `phi/options/contract.py`

Document what is already tracked per leg/position (quantity, side, premium, mark, Greeks) and what you still need for your target strategy class.

### 2) Enrich position state (if needed)

If your use case needs additional metadata (for example, entry-time snapshot Greeks or custom tags for multi-leg grouping), extend `phi/options/position.py` and keep additions backward compatible.

Recommended fields for research-grade runs:

- `option_type` (`call` / `put`)
- `strike`
- `expiration`
- `entry_price`, `exit_price`
- `delta`, `gamma`, `theta`, `vega`, `rho` snapshots
- `open_interest` and `implied_volatility` snapshots (when available)
- `strategy_id` / `group_id` for multi-leg bookkeeping

### 3) Add lifecycle handling in portfolio/equity curve logic

In `phi/backtest/portfolio.py` and/or `phi/backtest/options_engine.py`, ensure you handle:

- Multi-leg opening/closing as an atomic strategy intent.
- Mark-to-market using per-contract quotes.
- Expiration and assignment/exercise policy (or explicit simplification if unsupported).
- Separate attribution for premium P&L vs mark-to-market drift.

Keep assumptions explicit in logs and docs to avoid over-interpreting results.

### 4) Wire a deterministic options-only research pass

Before merging into full blended workflows, validate with a deterministic run:

1. Load normalized chain rows via `phi.options.data_adapter.fetch_options_data`.
2. Build strategy signals for one symbol/date window.
3. Simulate trades through one options engine path.
4. Export trade log + equity curve + risk metrics.

This gives you a controlled baseline for debugging leg selection and pricing behavior.

## 7.2 Extend the Streamlit workbench for options visualization

Phi-nance’s UI is modular under `app_streamlit/`. There are two good integration patterns:

- Add/expand options controls in `app_streamlit/live_workbench.py`.
- Add a dedicated page under `app_streamlit/pages/` if you want a separate options-research workspace.

### Suggested Options Explorer features

- Contract filters: symbol, expiry, strike range, option type.
- Chain table with bid/ask, IV, OI, and Greeks.
- Volatility smile chart (IV vs strike for selected expiry).
- Open-interest profile by strike.
- Greek timeseries for selected contract.
- Optional signal overlay (entry/exit markers) tied to a backtest run.

### Practical integration notes

- Use cached data loading where possible to avoid repeated DB/API hits.
- Keep plotting tolerant to sparse or partially missing Greek columns.
- Surface vendor/source metadata in UI so users know data provenance.

## 7.3 Add advanced signal engineering

With chain + Greek data available, add feature builders in your research pipeline and compose them into strategy rules.

### A) IV rank / relative volatility signal

Use rolling IV rank to detect cheap/expensive implied volatility regimes.

### B) Open-interest momentum

Track first differences (or smoothed deltas) in open interest by strike/expiry bucket.

### C) Gamma exposure (GEX)

Aggregate `gamma * open_interest * contract_multiplier` across the chain to estimate market-maker hedging pressure.

### D) Put-call pressure

Compute put/call volume (or OI) ratios as a sentiment feature.

### Signal composition

Blend options features with existing regime/blending layers:

- Implement feature transforms in your strategy research pipeline.
- Route the resulting scores through `phi/blending/blender.py` for portfolio-level coordination.
- Track each sub-signal in logs to preserve explainability.

## 7.4 Recommended implementation order

1. **Data reliability first:** validate normalized options chain quality and missing-field behavior.
2. **Engine correctness second:** verify lifecycle accounting (open, mark, close, expire).
3. **UI third:** add explorer views for contract and signal sanity checks.
4. **Model sophistication last:** layer IV/OI/GEX/put-call signals incrementally and benchmark each addition.

This sequencing reduces ambiguity when results shift and makes regressions much easier to isolate.

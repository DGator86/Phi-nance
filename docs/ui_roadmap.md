# UI roadmap (easy vs expert vs future)

This doc tracks **trading-UI patterns** we adopt over time. It is not a commitment schedule.

## Shipped (easy mode backtest)

| Pattern | Status |
|---------|--------|
| Price + equity **dual axis** | Done (`app_streamlit/easy_mode/backtest_screen.py`) |
| **Regime background** bands | Done (mapped semantic regimes) |
| **Trade markers** + hover **trigger** text | Done (`trade_events` from `phi.backtest.direct`) |
| **Context cards** (regime + 5d / 20d thrust) | Done (thrust = simple horizon returns; not multi-timeframe models) |
| Last-bar **indicator snapshot** | Done (`signal_snapshot_last` on results) |
| Three-preset **equity comparison** | Existing |
| **Multi-window regimes** (SHORT / MEDIUM / LONG k-means) | Done (`backtest_extras.train_multi_window_regimes`) |
| **Playback scrubber** (end index + truncated chart) | Done (`slice_replay_window`) |
| **Side-by-side preset compare** | Done (best vs selectbox) |
| **Robustness bootstrap** (shuffled returns → Sharpe) | Done (`bootstrap_sharpe_distribution`) |
| **RSI × threshold heatmap** | Done (`run_rsi_threshold_heatmap`) |
| **US daily close countdown** (ticker spotlight) | Done (`time_helpers`) |
| **Optional Dash wallboard** | Done (`app_dash/app.py`, `requirements-dash.txt`) |
| **Jupyter easy lab** | Done (`notebooks/10_easy_strategy_lab.ipynb`) |

## Backlog (higher effort)

| Idea | Notes |
|------|-------|
| **True multi-timeframe OHLCV** regimes | Resample bars per window instead of same bars + different k |
| **Intraday candle countdown** | Needs bar interval + exchange calendar |
| **PhiAI walk-forward one-click** | Wire existing PhiAI paths to easy-mode button |

## Principles

- **Transparency over black box** — show triggers and last-bar signal mix (AlgoFusion-style).
- **Avoid scope creep** — expert tuning stays in `expert_workbench.py`; easy mode stays read-mostly.
- **Streamlit limits** — heavy interactivity may later justify a **separate** Dash service, not a rewrite of `phi/`.

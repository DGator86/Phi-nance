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

## Backlog (higher effort)

| Idea | Notes |
|------|--------|
| **Playback scrubber** (time slider) | Needs Streamlit state + subset chart; consider `st.fragment` |
| **Side-by-side strategy replay** | Two result bundles same index; duplicate figure factory |
| **True LONG / MED / SHORT regimes** | Separate detectors or resampled OHLCV per timeframe |
| **Dash / real-time wallboard** | New app + deps; keep Streamlit as default |
| **Robustness one-click** | Wire PhiAI / walk-forward to a button |
| **Parameter heatmap** | Grid search + Plotly heatmap; cache results |
| **Candle countdown** | Only meaningful for intraday bars + live clock |

## Principles

- **Transparency over black box** — show triggers and last-bar signal mix (AlgoFusion-style).
- **Avoid scope creep** — expert tuning stays in `expert_workbench.py`; easy mode stays read-mostly.
- **Streamlit limits** — heavy interactivity may later justify a **separate** Dash service, not a rewrite of `phi/`.

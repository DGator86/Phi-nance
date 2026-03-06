# Repo status — where we are now

**As of:** current state vs BUILD_SPEC.md and the definition-of-done checklist.

---

## 1. Repo layout

```
Phi-nance/
├── .cursor/rules/
│   └── projection-only.mdc          # Lint rule: no strategy/routing/sizing
├── .env                              # Test keys (MASSIVE, TRADIER); do not commit
├── .env.example
├── .gitignore
├── BUILD_SPEC.md                     # Canonical phased plan
├── README.md
├── pyproject.toml
├── data/
│   ├── __init__.py
│   ├── bars/                         # data/bars/{ticker}/{year}.parquet (empty until backfill)
│   │   └── .gitkeep
│   └── providers/
│       ├── polygon_backfill.py      # Polygon/Massive 1m bars, 2y backfill
│       ├── tradier.py               # Live bars, chain, quotes, expirations
│       └── finra_short_volume.py    # Daily short volume (CDN, no key)
├── docs/
│   ├── DATA_AND_PHASE_STATUS.md
│   ├── GITHUB_REPOS.md
│   ├── GITHUB_DATA_SOURCES.md
│   ├── AGENT_WORLD_MODEL.md          # Using AWM with Phi-nance (MCP server, agent tasks)
│   ├── STRATEGY_LAB_GUI.md           # No-code GUI: strategy picker, backtest, Ask the Agent
│   ├── BACKTESTING_PY_INTEGRATION.md # backtesting.py (kernc) with our bar data + projection strategy
│   ├── LUMIBOT_INTEGRATION.md        # Lumibot backtest with our bar data + optional projection strategy
│   ├── REPO_STATUS.md                # This file
│   └── WHAT_I_NEED_FOR_PHASE2.md
├── fixtures/
│   └── projection_packet_sample.json
├── scripts/
│   ├── run_phase2.py                # Phase 2: backfill + live + FINRA + optional --check-gaps
│   ├── setup_data_spine.py          # Phase 2 bootstrap wrapper with retries, verification, sample fallback
│   ├── run_backtest.py              # WF backtest (synthetic or data/bars)
│   ├── run_paper_daily.py           # Phase 6.5: daily packets -> data/paper_packets/YYYY-MM-DD/
│   ├── run_dashboard.py            # Phase 6.5: 20d AUC, 75% cone, 20d IC, kill gate
│   ├── run_lumibot_backtest.py    # Lumibot backtest (pip install phi-nance[lumibot])
│   ├── run_backtesting_py.py      # backtesting.py/kernc (pip install phi-nance[backtesting])
│   ├── run_mcp_server.py          # MCP server for AWM (pip install phi-nance[mcp])
│   └── app.py                     # Strategy Lab GUI: streamlit run scripts/app.py (pip install phi-nance[gui])
├── src/phinence/
│   ├── __init__.py
│   ├── assignment/
│   │   ├── engine.py                # AssignmentEngine, 1m→5m resample, AssignedPacket
│   │   └── __init__.py
│   ├── composer/
│   │   ├── composer.py              # MFM → ProjectionPacket (stub calibration)
│   │   └── __init__.py
│   ├── contracts/
│   │   ├── projection_packet.py    # ProjectionPacket, Horizon, versioning, stub helpers
│   │   ├── assigned_packet.py      # AssignedPacket, coverage flags
│   │   └── __init__.py
│   ├── engines/
│   │   ├── liquidity.py            # POC/VAH/VAL, VWAP, swings
│   │   ├── regime.py               # ER + ATR% + EMA → regime probs
│   │   ├── sentiment.py            # RSI, trend, compression/expansion
│   │   ├── hedge.py                # EOD GEX/VEX (proshotv2 math)
│   │   ├── gex_math.py             # GEX/Vanna formulas
│   │   └── __init__.py
│   ├── backtesting_bridge/          # Optional: backtesting.py (kernc) using our bar store + projection
│   │   ├── data.py                 # bar_store_to_bt_df()
│   │   ├── strategy.py             # create_projection_strategy()
│   │   └── __init__.py
│   ├── gui/                         # Strategy Lab: run_backtest_for_strategy(), STRATEGY_CHOICES
│   │   ├── runner.py
│   │   └── __init__.py
│   ├── lumibot_bridge/              # Optional: Lumibot backtest using our bar store + projection
│   │   ├── data.py                 # bar_store_to_pandas_data()
│   │   ├── strategy.py             # create_projection_strategy_class()
│   │   └── __init__.py
│   ├── mfm/
│   │   ├── merger.py               # MarketFieldMap, build_mfm
│   │   └── __init__.py
│   ├── store/
│   │   ├── schemas.py              # BAR_1M_SCHEMA, BAR_5M_SCHEMA
│   │   ├── parquet_store.py        # ParquetBarStore, check_no_gap_more_than_n_bars
│   │   ├── memory_store.py        # InMemoryBarStore (backtests)
│   │   ├── bar_store_protocol.py
│   │   └── __init__.py
│   └── validation/
│       ├── walk_forward.py         # WF windows (intraday 3mo/2wk, daily 6mo/1mo)
│       ├── backtest_runner.py      # run_backtest_fold, AUC + cone coverage
│       ├── ablations.py            # ablation_threshold_met (0.02 AUC)
│       ├── paper_trading.py        # Kill criteria, paper_run_daily (bar_store), save/load packets
│       └── __init__.py
└── tests/
    ├── test_projection_packet.py
    ├── test_backtest.py
    └── test_gex_math.py
```

---

## 2. Definition-of-done checklist (BUILD_SPEC)

| # | Item | Status |
|---|------|--------|
| 1 | Repo skeleton: contracts/, data/providers/, store/, engines/, composer/, validation/ | ✅ Done |
| 2 | Parquet store + Arrow schemas; data/bars/{ticker}/{year}.parquet | ✅ Done |
| 3 | Polygon backfill script for 1m bars (2y) | ✅ Done (+ optional Massive client) |
| 4 | Tradier provider (live bars + chain snapshots) | ✅ Done |
| 5 | ProjectionPacket + fixtures + schema tests | ✅ Done |
| 6 | AssignmentEngine + coverage flags + AssignedPacket | ✅ Done |
| 7 | Liquidity Engine V1 (profile + swings) | ✅ Done |
| 8 | Regime Engine deterministic V1 | ✅ Done |
| 9 | Hedge Engine V1 (EOD dealer only) | ✅ Done |
| 10 | MFM integrator | ✅ Done |
| 11 | Composer V1 (drift/diffusion/cones) | ✅ Done (stub calibration; cones populated) |
| 12 | WF harness (intraday + daily windows) | ✅ Done |
| 13 | Ablation harness + >0.02 AUC threshold | ✅ Done |
| 14 | Phase 6.5 paper runner + kill criteria | ⚠️ Stub (kill logic + paper_run_daily stub; no live sandbox loop) |
| 15 | Projection-only lint rule | ✅ Done |

**Extra (not in original 15):** FINRA short volume provider + Phase 2 script + no-gap sanity check.

---

## 3. Phase-by-phase

| Phase | Spec | Status |
|-------|------|--------|
| **1** | ProjectionPacket boundary + projection-only rule | ✅ Done |
| **2** | Data spine: 2A historical bars, 2B live + chain, 2C FINRA | ✅ Automated bootstrap added via `python scripts/setup_data_spine.py` (uses Phase 2 + verification + sample fallback). |
| **3** | AssignmentEngine (router, 5m resample, coverage) | ✅ Done |
| **4** | Engines V1 (liquidity, regime, sentiment, hedge EOD) | ✅ Done |
| **5** | MFM (deterministic, replayable) | ✅ Done |
| **6** | Composer (drift/diffusion/cones) | ✅ Structure done; **calibration is stub** (direction ~1/3 each, drift 0). Real calibration = train on WF. |
| **6 gate** | Mean OOS AUC > 0.52 to proceed to paper | 🔲 Not run yet (need real data + calibrated composer) |
| **6.5** | Paper trading 8 weeks, dashboard, kill criteria | ⚠️ Kill logic + stub; **no automated sandbox loop or dashboard** yet. |
| **10** | Earn upgrades (intraday dealer, HMM, PrintStream) | 🔲 Later |

---

## 4. Data right now

- **data/bars/** — Empty in repo (parquet gitignored). Populated when you run `run_phase2` with valid MASSIVE/Polygon key.
- **data/short_volume/** — Not in repo (parquet gitignored). Populated when you run `run_phase2` (FINRA needs no key).
- **.env** — Present with test MASSIVE + TRADIER keys for Phase 2 runs.

---

## 5. What’s next (in order)

1. **Run data spine setup**  
   `python scripts/setup_data_spine.py --tickers SPY QQQ --years 2` (optionally `--sample-only`). This wraps Phase 2, retries failures, and validates no-gap checks.

2. **Composer calibration**  
   Replace stub direction/drift with something trained on WF train windows (e.g. simple model or rules using MFM features).

3. **Run WF and hit Phase 6 gate**  
   Use `scripts/run_backtest.py` or validation harness on real data; get mean OOS AUC; aim for > 0.52.

4. **Phase 6.5**  
   Implement Tradier sandbox loop (daily run, persist packets, compute 20d AUC / 75% cone / 20d IC) and a minimal dashboard or log summary.

---

## 6. Summary

- **Built:** Skeleton, contracts, store, providers (Polygon/Massive, Tradier, FINRA), all four engines, MFM, Composer, WF + ablation, paper kill criteria, projection-only rule. Backtests run on synthetic data.
- **Not yet done:** Composer calibration, passing the Phase 6 AUC gate, and a real Phase 6.5 paper-trading loop + dashboard.

So: **implementation is in place through Phase 6 (with stub calibration); Phase 2 bootstrapping is now automated, and Phase 6.5 automation remains a next concrete step.**

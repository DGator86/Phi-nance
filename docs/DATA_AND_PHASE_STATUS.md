# Data Sources & Next Steps (per BUILD_SPEC.md)

## Data sources: what’s missing

| Source | Spec | Status | Notes |
|--------|------|--------|--------|
| **Historical 1m bars (2+ years)** | Phase 2A — Polygon or equivalent | **Wiring done; data not populated** | `data/providers/polygon_backfill.py` and optional `fetch_1m_bars_massive()` exist. You must run backfill with `POLYGON_API_KEY` (or Massive key). `data/bars/` is still empty until then. |
| **Live 1m bars + options chain** | Phase 2B — Tradier | **Implemented** | `data/providers/tradier.py`: `fetch_live_1m_bars`, `fetch_chain_snapshot`, `get_quote`, `options_expirations`. Staleness via `chain_snapshot_staleness_seconds`. |
| **FINRA short volume** | Phase 2C — daily | **Implemented** | `data/providers/finra_short_volume.py` — public CDN, no key. Persists to `data/short_volume/YYYYMMDD.parquet`. |

**Summary**

- **Missing data sources:** (1) **Populated** historical 1m bars (run Polygon/Massive backfill with your API key). (2) **FINRA short volume** (optional; no code yet).
- **Not missing:** Tradier live bars, chain snapshots, quotes, expirations. Polygon/Massive backfill **code** is in place.

---

## Next steps per phase document

Follow the **Definition-of-done checklist** at the bottom of BUILD_SPEC.md and the **“Done when”** criteria for each phase. Suggested order:

### 1. Phase 2 — Data spine (finish)

- [ ] **Run Polygon/Massive backfill** for your universe so `data/bars/{ticker}/{year}.parquet` exists for 2+ years.
- [ ] **Verify Phase 2A done when:** Whole universe has continuous RTH 1m sequences; “no-gap >5 bars” sanity checks pass (e.g. use `store.parquet_store.check_no_gap_more_than_n_bars`).
- [ ] **Verify Phase 2B done when:** “Today’s live stream” from Tradier persists into the same schema as historical (ingest path + schema alignment).
- [ ] *(Optional)* Add **FINRA short volume** (Phase 2C): daily pull, store, and any wiring into MFM/engines if you want it in V1.

### 2. Phase 4 — Engines “done when” (validate)

- [ ] **4A Liquidity:** Compare levels (POC/VAH/VAL, swings, VWAP) to TradingView on a few tickers/dates; tune if needed.
- [ ] **4B Regime:** Spot-check random windows; confirm confidence behaves sensibly.
- [ ] **4C Sentiment:** Confirm RSI/trend/compression match a reference implementation.
- [ ] **4D Hedge:** Confirm stable EOD dealer landmarks (GEX/VEX, zero-gamma zone if you add it) and EPP scalar.

### 3. Phase 5 — MFM

- [ ] Already implemented: `MarketFieldMap` + `build_mfm`. Confirm **done when:** same inputs → same MFM (deterministic, replayable).

### 4. Phase 6 — Composer + gate

- [ ] **Composer calibration:** Replace stub direction/drift/cones with real calibration (e.g. trained on WF train windows).
- [ ] **Done when:** Cones never inverted; packet fully populated for all horizons (confidence may be low).
- [ ] **Gate to 6.5:** Mean OOS AUC > 0.52 on WF before starting paper trading.

### 5. Phase 6.5 — Paper trading

- [ ] Implement **Tradier sandbox runner** that emits daily `ProjectionPacket` for the full universe (stub exists in `validation/paper_trading.py`).
- [ ] Add **rolling dashboard:** 20d AUC, 75% cone coverage, 20d IC, regime distribution.
- [ ] Enforce **kill criteria:** AUC < 0.50 for 10 days; 75% cone < 60% or > 90%; IC negative 2+ weeks.
- [ ] **Done when:** 8 weeks in-range.

### 6. Checklist check-off (BUILD_SPEC.md bottom)

- [ ] Mark items **1–15** done as you complete them (skeleton, store, backfill script, Tradier, ProjectionPacket, AssignmentEngine, engines, MFM, Composer, WF harness, ablation, paper runner, projection-only rule).

---

## One-line summary

**Missing data:** (1) Historical 1m bars **in the store** (run backfill); (2) FINRA short volume (optional, not built).  
**Next steps:** Run backfill → validate Phase 2 “done when” → validate Phase 4 “done when” → calibrate Composer → pass Phase 6 gate (OOS AUC > 0.52) → implement Phase 6.5 paper runner and run 8 weeks.

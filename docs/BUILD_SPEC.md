# EPP v2 — Phased Build Spec (Canonical)

**Principle:** Projection system ends at `ProjectionPacket`. No strategy selection, routing, or sizing. Validate *accuracy* before *profitability*.

---

## Hard boundaries

| Boundary | Rule |
|----------|------|
| **Projection-only** | Code review: no strategy selection, no routing, no sizing inside projection pipeline |
| **Dealer fields** | EOD context until proven; `dealer_field_frequency: "eod"` in V1 |
| **Prints** | Stubbed; bars before prints; activate only if sub-minute horizons required |
| **Data budget** | Polygon (or equivalent) required for 2+ years 1m bars; Tradier alone insufficient |
| **Trust gate** | Phase 6.5 paper trading (8 weeks) mandatory between WF and any live capital |

---

## Phase 1 — Lock the boundary: projection ends at `ProjectionPacket`

**Deliverables**

- `ProjectionPacket` Pydantic model + versioning + sample JSON fixtures
- “Projection-only” lint rule in code review

**Done when**

- You can generate a valid `ProjectionPacket` with all required fields; confidences may be `0.0` when data is missing.

---

## Phase 2 — Data spine (Phase 6 cannot run without it)

### 2A) Historical 1m bars (backtest fuel)

- **Polygon (or equivalent)** for **2+ years** of 1-minute bars
- Parquet layout: `data/bars/{ticker}/{year}.parquet`

**Done when**

- Whole universe has continuous RTH sequences; “no-gap >5 bars” sanity checks pass.

### 2B) Live bars + live options (Tradier)

- Tradier: live 1m bars + chain snapshots
- Snapshot staleness tracking per symbol

**Done when**

- “Today’s live stream” persists into same schema as historical.

### 2C) Optional — FINRA short volume

- Daily pull only; not blocking V1.

---

## Phase 3 — Assignment Engine (strict router, not processor)

**Deliverables**

- `AssignedPacket` + coverage flags + warnings
- Derive 5m bars from 1m (resampling) inside router

**Done when**

- Every engine receives typed bundles; missing data never crashes—coverage drops, confidence → 0.

---

## Phase 4 — Primary engines V1 (terrain only)

### 4A) Liquidity Engine (bar-only)

- Volume profile: POC, VAH, VAL, HVN, LVN
- Structural levels: swings, gaps, unfilled ranges, anchored VWAP

**Done when**

- Levels “close enough” to TradingView across tickers/dates.

### 4B) Regime Engine (deterministic V1)

- ER + ATR% + EMA alignment → soft regime probabilities

**Done when**

- Random sampled windows “look right”; confidence behaves sensibly.

### 4C) Sentiment Engine (minimal)

- RSI + trend alignment + compression/expansion

**Done when**

- Indicators match a reference implementation.

### 4D) Hedge Engine — EOD dealer fields first

- EOD snapshot → GEX profile + derived phi + EPP proxy (calibrated later)
- **No** intraday vanna/charm steering at hobbyist refresh rates in V1

**Done when**

- Stable EOD dealer landmarks (walls, zero-gamma zone) + EPP scalar.

---

## Phase 5 — MFM integration

**Deliverables**

- `MarketFieldMap` merger: namespaced fields + landmarks + steering vectors

**Done when**

- MFM is deterministic and replayable (same inputs → same MFM).

---

## Phase 6 — Composer: drift + diffusion + cones

**Core outputs per horizon**

- Direction distribution (up/down/flat)
- Drift in bps
- Vol cones (50/75/90) + annualized σ

**Done when**

- Cones never inverted; packet fully populated for all horizons (confidence may be low).

**Gate to Phase 6.5**

- Mean OOS AUC > 0.52 to proceed to paper trading.

---

## Phase 6.5 — Paper trading validation (8 weeks, mandatory)

**Deliverables**

- Tradier sandbox runner emitting daily `ProjectionPacket` for full universe
- Rolling dashboard: 20d AUC, 75% cone coverage, 20d IC, regime distribution

**Kill criteria**

- AUC < 0.50 for 10 consecutive days
- 75% cone coverage < 60% or > 90%
- IC negative for 2+ weeks

**Done when**

- 8 weeks in-range.

---

## Walk-forward — horizon-specific

| Signal type | Train | Test | Step | Window |
|-------------|-------|------|------|--------|
| **Intraday** (1m/5m) | ~3 months (~63 td) | ~2 weeks (~10 td) | 2 weeks | Expanding default |
| **Daily** (EOD dealer) | ~6 months (~126 td) | ~1 month (~21 td) | 1 month | Expanding |

Purge + embargo at boundaries.

---

## Ablations — “earns complexity” rule

**Baseline OOS metrics (per fold + averaged)**

- **Directional AUC** (primary)
- **Cone coverage calibration** at 50/75/90 (primary)
- Optional: IC, DSR later

**Ablation rule (V1)**

- Engine stays only if it improves **mean OOS AUC by > 0.02** (2 pp) *or* materially improves cone calibration without hurting AUC.

---

## Phase 10 — Earn upgrades (only after gates pass)

Order:

1. **Intraday dealer refresh** (after EOD dealer proves ablation value)
2. **HMM regime** (after deterministic regime proves useful)
3. **PrintStream activation** (only with credible prints/NBBO and sub-minute horizons)

---

## Definition-of-done checklist (build tracker)

Work linearly, phase by phase.

- [ ] **1** Repo skeleton: `contracts/`, `data/providers/`, `store/`, `engines/`, `composer/`, `validation/`
- [ ] **2** Parquet store + Arrow schemas; `data/bars/{ticker}/{year}.parquet`
- [ ] **3** Polygon backfill script for 1m bars (2y)
- [ ] **4** Tradier provider (live bars + chain snapshots)
- [ ] **5** `ProjectionPacket` + fixtures + schema tests
- [ ] **6** AssignmentEngine + coverage flags + `AssignedPacket`
- [ ] **7** Liquidity Engine V1 (profile + swings)
- [ ] **8** Regime Engine deterministic V1
- [ ] **9** Hedge Engine V1 (EOD dealer only)
- [ ] **10** MFM integrator
- [ ] **11** Composer V1 (drift/diffusion/cones)
- [ ] **12** WF harness (intraday + daily windows)
- [ ] **13** Ablation harness + >0.02 AUC threshold
- [ ] **14** Phase 6.5 paper runner + kill criteria
- [ ] **15** Projection-only lint rule

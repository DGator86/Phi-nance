# Roadmap: options signals + MTF + regime + info-theory + ML tuning

This maps your five goals to **what Phi-nance already has** and **what still needs to be built** so the product feels like one system—not scattered tabs.

## 1. Trade signals for options (entry, target exit, stop)

**Already in the codebase**

- `phi/options/engine.py` — structures (long call/put, spreads, etc.), entry on signal, exits: **profit target**, **stop loss**, DTE exit, expiry.
- `phi/options/simulator.py` — `profit_exit_pct`, `stop_exit_pct`, signal threshold, structure, DTE.
- `phi/options/ai_advisor.py` — LLM JSON: structure, direction, `entry_signal` (ENTER/WAIT/SKIP), sizing—not numeric strikes/targets/stops.

**Gap**

- No single **canonical “signal card”** object exposed to UI/agents: e.g. `{ action, structure, strikes_or_rules, entry_px_band, target_px_or_pct, stop_px_or_pct, horizon_dte, rationale_id }`.
- Easy mode is equity-focused; **Trading desk** + expert workbench hold options pieces but don’t emit one consolidated daily/scan output.

**Build direction**

- Define a small **Pydantic/dataclass schema** (e.g. `phi/options/signal_card.py`) generated from: composite signal + playbook row + optional chain snapshot.
- One Streamlit panel or API endpoint: “today’s playbook-aligned suggestion” with explicit **target/stop in % or $**.

## 2. MTF confirmation

**Already**

- `phi/regime/mtf_matrix.py` — resampled-bar regimes + confluence score (daily easy mode gets 1D/1W/1ME-style columns; intraday data unlocks finer TFs).
- Easy mode: three-window **feature** stack (12/20/40) ≠ calendar TFs; both are labeled in `docs/regime_timeframe_matrix.md`.

**Gap**

- MTF is not yet a **hard gate** on options signals (e.g. “only ENTER if weekly + daily confluence > X and 4H not bearish”).
- No single **MTF badge** on the options signal card.

**Build direction**

- Add optional `mtf_gates: dict` to signal generation: min confluence, forbidden semantic combos, weights by TF role (bias vs trigger).

## 3. Regime-aware reasoning

**Already**

- `phi/options/regime_playbook.py` + `docs/options_regime_playbook.md` — regime → structures, DTE/delta bands, transitions.
- `phi/regime` trainable detectors + regime-weighted blending in `phi/backtest/direct.py`.
- `regime_engine/` — full taxonomy / species / probabilities (expert path).

**Gap**

- **Human-readable reasoning** is fragmented (playbook row vs detector label vs species)—not one paragraph or bullet list tied to the signal card.
- `phi/options/ai_advisor.py` can reason in JSON but isn’t wired as the default “explain this trade.”

**Build direction**

- Template renderer: given `regime_label`, `playbook_row`, `mtf_snapshot`, produce a fixed **Reasoning** section (deterministic first; LLM optional overlay).

## 4. Information-theoretic processing

**Already**

- Indicators in the workbench catalog: entropy, mutual information, Fisher, KL-style features (used in the big auto stack in easy mode).
- These feed **composite signal** and options engine **indirectly** via the same pipeline as other indicators.

**Gap**

- Info-theory features are not **surfaced** on the options signal (e.g. “entropy spike → favor shorter hold / reduce size”).
- No dedicated **info regime** dimension in the playbook (could be added as a feature input to mapping).

**Build direction**

- Expose last-bar info metrics on the signal card and optional **rules** in playbook JSON (thresholds → sizing or structure tweak).

## 5. Constant parameter tuning (AI & ML)

**Already**

- `phi/phiai/auto_tune.py` — Optuna, TPE/NSGAII, **walk-forward** scoring, `run_phiai_optimization`.
- Env knobs: `PHIAI_*` in config; expert UI hooks for training.
- `regime_engine/param_tuner.py` exists for engine-side tuning.

**Gap**

- Tuning is **batch/on-demand**, not a documented **continuous loop** (schedule → new params → promote → signal card uses promoted params).
- No single “promotion” path from PhiAI best trial → options signal defaults.

**Build direction**

- Nightly or manual job: run PhiAI on a frozen dataset id → write `runs/best_params/<id>.json` → signal card and backtest read that file when `USE_PROMOTED_PARAMS=1`.

---

## Suggested build order (vertical slices)

| Phase | Deliverable | Touches |
|-------|-------------|---------|
| **A** | `OptionsSignalCard` schema + generator from OHLCV + playbook + composite signal | **Done:** `phi/options/signal_card.py`, `signal_generator.py`, `tests/test_options_signal_card.py` |
| **B** | Trading desk panel: one card with entry / target% / stop% / DTE | **Done:** `app_streamlit/trading_desk.py` — **Build signal** button |
| **C** | Wire `mtf_matrix` + confluence into generator as optional filters | `phi/regime/mtf_matrix.py`, generator |
| **D** | Reasoning string + last-bar info metrics on card | playbook + indicators snapshot |
| **E** | PhiAI promotion file + env to load promoted params in generator | `phi/phiai/`, docs |

---

## How this differs from “easy automatic backtest”

Easy mode optimizes **equity Sharpe** on a huge stacked signal book. Your list is **options-first**, **explainable**, **MTF-gated**, and **continuously tuned**. That’s a different top-level workflow; the roadmap above connects existing engines instead of replacing them.

If you want implementation next, say which phase (**A–E**) to start with; **A+B** gives the fastest visible “this is what I meant” demo.

# Options Phase 3

Phase 3 adds a polished options workbench UI, advanced strategy templates, historical-IV fallback support, early-exercise heuristics for American options, and RL-facing integration hooks.

## Highlights

- Streamlit options panel now includes dynamic strategy controls, expiry modes, real-time Greeks/premium, and payoff chart.
- New advanced strategies: butterfly, calendar, diagonal, covered call, protective put, collar.
- `HistoricalIVSurface` supports nearest cached IV snapshot by date, with historical-volatility fallback when snapshots are missing.
- Early-exercise heuristic (`phi/options/early_exercise.py`) supports American-option simulation triggers.
- Options engine improvements include leg/date loop optimization and richer leg-level trade log events.
- RL agent hooks:
  - Strategy R&D now includes option templates in fallback template library.
  - Risk monitor emits explicit options hedge actions (`buy_put_protection`) when profile hedge ratio > 0.

## Usage notes

- Historical options data availability is provider-limited. Without provider snapshots, the system falls back to HV-derived IV.
- Early exercise is heuristic, not a full LSM engine.
- Covered/protective/collar strategy classes model option legs only; underlying stock leg accounting remains at portfolio layer.

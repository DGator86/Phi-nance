# Phi-nance — EPP v2

Projection system whose **hard boundary** is `ProjectionPacket`. No strategy selection, routing, or sizing inside the pipeline.

---

## Get started (Strategy Lab GUI)

**First time setup:**

1. Open a terminal in this folder (where you see `scripts`, `src`, `pyproject.toml`).
2. Run: **`pip install -e '.[gui]'`** (one line only; use single quotes in bash/Linux).
3. Run: **`streamlit run scripts/app.py`** (or `python3 -m streamlit run scripts/app.py` if streamlit not found).
4. Your browser will open to **Phi-nance Strategy Lab** — pick strategies, run backtests, compare results, and ask the agent.

**Full step-by-step and troubleshooting:** [SETUP.md](SETUP.md)

**Windows:** After setup, you can double-click **`run_strategy_lab.bat`** to start the app.

---

## More

- **Spec:** [BUILD_SPEC.md](BUILD_SPEC.md) — canonical phased build and definition-of-done.
- **Data:** Polygon (or equivalent) for 2+ years 1m bars; Tradier for live bars + chain snapshots.
- **Validation:** Walk-forward (horizon-specific) → Phase 6.5 paper trading (8 weeks) before any trust.

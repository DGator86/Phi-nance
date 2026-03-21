# Repository layout (where things live)

This is the **practical map** of the Phi-nance monorepo. Use it when deciding where to add code or documentation.

## Top-level packages

| Path | Role |
|------|------|
| **`phi/`** | Primary **runtime library**: config, logging, data/cache/fetchers, indicators, blending, regime detection & training, equity/options backtesting (`phi.backtest`), options adapters, validation, PhiAI hooks. **Most new engine logic belongs here.** |
| **`phinance/`** | **Research / agents / live glue**: orchestration, RL hooks, ecosystem adapters (Lumibot, TensorTrade, agent-cli), live trading helpers. Overlaps conceptually with `phi` but targets higher-level workflows and integrations—not a duplicate of the same modules. |
| **`regime_engine/`** | **Market Field Theory (MFT)** pipeline: field computations, projections, regime features used by research and notebooks. Consumed by `phi` / notebooks; not a second “app” package. |
| **`strategies/`** | **Lumibot-style strategy classes** and blended workbench strategy referenced by backtests and legacy Streamlit paths. Keep strategy entrypoints here when they are class-based adapters. |
| **`app_streamlit/`** | **Streamlit UI**: `main.py` (easy mode default), `expert_workbench.py`, pages, handlers, config. See [`easy_mode.md`](easy_mode.md). |
| **`legacy/`** | **Deprecated** Lumibot/dashboard Streamlit apps and older GUIs. Referenced only in **docs** and fallback scripts (e.g. `scripts/run_streamlit.sh`). **No `import legacy` in current `phi/` or `app_streamlit/` core paths**—do not build new features here. |
| **`notebooks/`** | Jupyter entrypoints; use `notebook_setup.py` / `00_getting_started.ipynb` for `sys.path` and `.env`. |
| **`scripts/`** | CLIs: data spine, backtests, engine health, deploy helpers. |
| **`tests/`** | `pytest` tree; mirror package boundaries (`phi`, `phinance`, `app_streamlit`) where possible. |
| **`docs/`** | User and contributor documentation (this file, architecture, options, deployment). |
| **`frontend/`** | Secondary / experimental UI assets (not the main Streamlit app). |

## Data flow (simplified)

```text
Vendor APIs / cache  →  phi.data.*  →  OHLCV DataFrame
                              ↓
phi.indicators + phi.blending  →  composite signal
                              ↓
phi.regime (+ regime_engine for MFT)  →  regime labels / features
                              ↓
phi.backtest / phi.options  →  metrics, equity curves
                              ↓
app_streamlit / notebooks / scripts  →  humans & automation
```

## When to touch what

- **New indicator or blend rule** → `phi/indicators`, `phi/blending`.
- **Regime model or k-means/HMM wiring** → `phi/regime`; MFT-specific math → `regime_engine/` or `phi` MFT indicators as appropriate.
- **Streamlit screen or form** → `app_streamlit/` (prefer `easy_mode/` for layperson flows, `expert_workbench.py` for full workbench).
- **Agent / broker bridge** → `phinance/` (check existing adapters first).
- **User strategy class for Lumibot** → `strategies/`.

## Legacy folder

Treat **`legacy/`** as **read-only archive**. Prefer `app_streamlit/main.py` or `expert_workbench.py`. Older docs that mention `streamlit run legacy/dashboard.py` are historical; update them to the modular app when you edit those pages.

## Related docs

- High-level architecture: [`Architecture.md`](Architecture.md)
- Easy mode UI: [`easy_mode.md`](easy_mode.md)
- Dev branching: [`DEV_WORKFLOW.md`](DEV_WORKFLOW.md)

# Phi-nance — stakeholder overview, SWOT, Start–Stop–Continue

High-level introduction for stakeholders and new team members. Deeper technical maps: [architecture_layout.md](architecture_layout.md), [signals_research_roadmap.md](signals_research_roadmap.md), [alpaca_paper_options.md](alpaca_paper_options.md).

---

## 1. Outline of main features

**Phi-nance** is a quantitative trading research platform for building, blending, and validating regime-aware strategies. Core capabilities:

| Area | Description |
|------|-------------|
| **Regime-aware MFT engine** | Multi-factor timing (MFT) regime detection, feature engineering, and transition maps (`regime_engine/`, `phi/regime/`). |
| **Multi-vendor data spine** | Unified caching for OHLCV (e.g. **Unusual Whales → yfinance** fallback, Polygon/Massive, Alpha Vantage aliases, PostgreSQL for options rows). Extensible via fetcher registry. |
| **Indicator catalog** | Technical and information-theoretic indicators with configurable parameters (`phi/indicators/`). |
| **Signal blending** | Weighted sum, voting, or regime-weighted combination (`phi/blending/`). |
| **PhiAI optimization** | Automated parameter tuning (e.g. Optuna) with walk-forward validation (`phi/phiai/`). |
| **Backtesting** | Direct backtest paths (`phi/backtest/`); ecosystem notes for Lumibot / TensorTrade / agent-cli in [ecosystem_integration.md](ecosystem_integration.md). |
| **Options playbook & signals** | Regime-tagged playbook, structured **options signal card** (entry / target / stop, MTF, UW chain when keyed); export JSON for QuantConnect research. |
| **QuantConnect bridge** | `quantconnect/` Lean template, `scripts/export_quantconnect_bundle.py`, headless `phi.api.qc_export` on port 8080. |
| **CLI & scripts** | Data spine, backtests, engine health, Alpaca paper health + MTF confirm, etc. (`scripts/`). |

Configuration uses environment variables (`.env`), local data cache, and structured run directories.

---

## 2. SWOT analysis

| **Strengths** | **Weaknesses** |
|---------------|----------------|
| Modular layout — `phi/`, `phinance/`, `regime_engine/`, `quantconnect/` separated. | Lean cloud is the primary execution surface; local stack is research + export. |
| Extensible data spine — new vendors plug into cache fetchers. | No first-party REST API; external automation relies on scripts or embedding Python. |
| Caching — reduces repeat vendor calls. | Streaming / low-latency data is not a core focus; live paths are broker- and script-shaped. |
| Regime-aware blending and playbooks. | Onboarding: multiple optional `requirements-*.txt` files and env vars. |
| PhiAI / optimization built in. | Advanced agent paths vary in documentation depth. |
| Ecosystem adapters documented for downstream tools. | Test suite is broad but live broker / options order paths need more coverage. |

| **Opportunities** | **Threats** |
|-------------------|-------------|
| Headless **FastAPI** (or similar) service over core flows. | Vendor API changes (data brokers, Alpaca). |
| Deeper QuantConnect or other cloud backtest brokers. | Competing platforms with turnkey cloud + execution. |
| Expand live execution (Alpaca options orders, IBKR, etc.). | Dependency matrix (ML, optional stacks) and upgrade churn. |
| Containerized deploy (Docker) for 24/7 jobs. | Secret handling in `.env` — process and rotation discipline. |
| Clearer docs and notebooks → contributors. | Model / regime assumptions can decay vs market structure. |

---

## 3. Start — Stop — Continue

### Start doing

- **Headless service** — FastAPI (or equivalent) over selected flows: OHLCV fetch, signal card, backtest submit, status polling.
- **Structured production logging** — JSON logs for fetch/backtest/job boundaries where a service exists.
- **Targeted tests** — Critical paths: cache fetch, blend pipeline, `run_direct_backtest` invariants, Alpaca paper smoke.
- **Deployment artifacts** — Dockerfile / compose with env-only secrets.
- **Optional** — QuantConnect or additional vendors where ROI is clear.

### Stop doing

- **Leaking presentation into core** — Keep `phi/` and `phinance/` free of web-framework imports; core stays importable for scripts, notebooks, and QC-side ports.
- **Opaque global script state** — Prefer explicit parameters and small CLIs for automation.
- **One-off manual setup only** — Improve single-path bootstrap docs / optional `make` or `uv` recipes over time.
- **Blocking HTTP on long jobs** — Use async jobs or a worker queue for heavy backtests (future).

### Continue doing

- **Modularity** — Data, indicators, blending, backtest as composable layers.
- **Env-based config** — `.env` for secrets; never commit real keys.
- **Caching** — File-backed cache for OHLCV and runs.
- **Architecture docs** — Keep `docs/architecture_layout.md` and workflow docs current.
- **Multiple backtest surfaces** — Direct engine plus documented external engines where used.

---

## 4. Suggested immediate direction

Prioritize a **small service or job runner** that can: load config + env, pull OHLCV (existing cache path), run a defined backtest or emit a signal artifact, and log results — without requiring Streamlit. That unlocks automation and safer iteration toward live execution.

---

*Document version: maintained in-repo; revise as the product evolves.*

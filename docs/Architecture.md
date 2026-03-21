# Phi-nance Architecture

For a **folder-by-folder map** (phi vs phinance vs legacy, where to add code), see [`architecture_layout.md`](architecture_layout.md).

## High-Level Overview

Phi-nance is organized as a modular research platform with clear separation between data access, strategy logic, optimization, and user interfaces.

Primary layers:

1. **Data layer** (`phi.data`, `data/providers`, `phinance/data`): vendor adapters, cache, and staleness-aware retrieval.
2. **Strategy and signal layer** (`strategies`, `phi.indicators`, `regime_engine`): indicator generation, MFT regime features, and strategy outputs.
3. **Blending/optimization layer** (`phi.blending`, `phi.phiai`, `phinance/optimization`): weighted/voting/regime-aware blending and auto-tuning workflows.
4. **Backtesting layer** (`scripts/run_backtest.py`, `phi.options.backtest`): equities/options simulation and result emission.
5. **Presentation layer** (`app_streamlit`, `legacy/dashboard.py`): modular Streamlit workbench and legacy dashboard.

## Module Map

| Module/Path | Responsibility |
|---|---|
| `phi/config.py` | Centralized environment-backed runtime settings (paths, logging level, PhiAI defaults). |
| `phi/logging.py` | Shared logger setup, level resolution, file+console handler wiring. |
| `phi/exceptions.py` + `phinance/exceptions.py` | Typed exception hierarchy used for predictable error handling. |
| `phi/utils/validation.py` | Input sanitization and validation primitives used by scripts/UI paths. |
| `phi/data/` | Cache and vendor fetch abstraction for OHLCV/options-adjacent data retrieval. |
| `phi/indicators/` | Indicator registry and computation helpers. |
| `phi/blending/` | Signal-combination methods (`weighted_sum`, `voting`, `regime_weighted`). |
| `phi/phiai/` + `phinance/optimization/` | Parameter search, orchestration, walk-forward optimization, and explainability helpers. |
| `phi/options/` | Options pricing/backtesting support, strategy primitives, and Greeks utilities. |
| `regime_engine/` | Market Field Theory (MFT) features, taxonomy, regime scoring, and tuning components. |
| `app_streamlit/` | Modular Streamlit app entry points, pages, UI state, and workflow orchestration. |
| `tests/` | Unit/extended test coverage for blending, data, options, app handlers, and optimizers. |

## Data Flow

Typical research flow:

1. **Dataset setup** (`scripts/setup_data_spine.py`) prepares and validates bar/short-volume datasets.
2. **Data retrieval** (`phi.data.cache.fetch_and_cache`) resolves vendor data and stores cache artifacts.
3. **Feature generation** computes indicators and regime descriptors (`phi.indicators`, `regime_engine`).
4. **Signal blending** combines strategy outputs (`phi.blending.blender.blend_signals`).
5. **Optimization (optional)** runs PhiAI/optimizer workflows for parameter selection and walk-forward robustness.
6. **Backtest execution** runs equity/options simulations and persists outputs into run directories.
7. **UI/analysis** surfaces metrics and artifacts in Streamlit modules and generated reports.

## Configuration Management

`phi/config.py` exposes a `Settings` dataclass populated from environment variables.

- Paths: `DATA_CACHE_DIR`, `RUNS_DIR`, `LOGS_DIR`.
- Runtime flags: `LOG_LEVEL`, `DEBUG`.
- Optimization defaults: `PHIAI_DEFAULT_N_TRIALS`, `PHIAI_PARALLEL_JOBS`, `PHIAI_WALK_FORWARD_WINDOWS`.
- Compatibility alias: `DATA_CACHE_ROOT` remains available for backward compatibility.

The settings object is used throughout runtime modules to reduce scattered path logic and centralize defaults.

## Logging and Error Handling

- `phi/logging.py` standardizes logger setup and output format.
- Production modules should use logger calls instead of `print` for operational events.
- Custom exceptions from `phi.exceptions` / `phinance.exceptions` are preferred over broad `Exception` raises.
- Validation helpers in `phi.utils.validation` should be used to fail early with clear error messages.

## Testing and CI Strategy

- Tests live under `tests/` and include unit-style and feature-level checks.
- Default local command: `pytest` (see `pyproject.toml` for configured options).
- CI expectations: lint/type-check/test gates, coverage reporting, and docs synchronization for user-facing changes.

## UI Architecture

The Streamlit workbench (`app_streamlit/main.py`) coordinates:

- session state and cache helpers,
- page-level modules under `app_streamlit/pages/`,
- shared UI components and rendering helpers,
- backtest controls and run result display.

Legacy dashboards remain available for backward compatibility and experimentation.

# Indicators

## Existing indicator support

Indicators are registered via the indicator registry/spec structures and surfaced in Streamlit controls.

## Adding a new indicator

1. Implement indicator computation in `phi/indicators/` (or connected strategy module).
2. Register it in the indicator registry (`phi/indicators/registry.py`).
3. Add UI parameter metadata to `app_streamlit/config.py` if it should be tunable in the workbench.
4. Add/extend tests in `tests/` for calculation and signal expectations.
5. Update docs and quick examples.

## Parameter grids and PhiAI

For auto-tuning support, define parameter bounds/ranges consumed by optimization routines.

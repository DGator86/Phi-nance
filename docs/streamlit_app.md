# Streamlit App

## Entry points

- `app_streamlit/main.py` (default: **easy mode** — three-screen layperson UI)
- `app_streamlit/expert_workbench.py` (full modular workbench)
- `legacy/dashboard.py` (legacy app)

## Design

The Streamlit UI is modularized with:

- page modules in `app_streamlit/pages/`
- reusable components in `app_streamlit/ui_components.py` and related modules
- shared app state helpers in `app_streamlit/state.py`

## State management

Session-state values coordinate selected datasets, indicator settings, blend method, and run outputs.

## Customization

- Add new page modules under `app_streamlit/pages/`.
- Extend controls by updating page handlers/config specs.
- Keep user-facing errors routed through centralized exception + logger patterns.

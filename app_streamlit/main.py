"""Default Streamlit entry: layperson-friendly Phi-nance (easy mode).

For the full workbench: `streamlit run app_streamlit/expert_workbench.py`
"""

from __future__ import annotations

from app_streamlit.easy_mode.app import run_easy_app


def main() -> None:
    run_easy_app()


if __name__ == "__main__":
    main()

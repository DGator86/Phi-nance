"""
Bootstrap Phi-nance for Jupyter / JupyterLab.

In any notebook, run the first cell from the docs (runpy + this file), or from a
terminal:

    python notebook_setup.py

That adds the repo root to sys.path, sets IS_BACKTESTING for lumibot, and loads .env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the clone root (contains ``regime_engine/`` and ``phinance/``)."""
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p / "regime_engine").is_dir() and (p / "phinance").is_dir():
            return p
    raise RuntimeError(
        "Could not find Phi-nance project root (need regime_engine/ and phinance/). "
        "In JupyterLab use File → Open Folder on the repo, or start the server from "
        "the repository directory."
    )


def setup(start: Path | None = None) -> Path:
    root = find_project_root(start)
    rs = str(root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    os.environ.setdefault("IS_BACKTESTING", "True")
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass
    return root


ROOT = setup()

if __name__ == "__main__":
    print(f"Phi-nance path ready: {ROOT}")

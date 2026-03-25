#!/usr/bin/env python3
"""Lightweight import smoke test. Run from repo root with venv active.

Use after clone or when debugging ``ModuleNotFoundError`` / wrong interpreter.
For full MFT pipeline validation, use: python scripts/engine_health.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    modules = [
        "phi.backtest.direct",
        "phi.blending.blender",
        "phi.api.qc_export",
        "regime_engine",
        "phinance",
    ]
    failed: list[str] = []
    for name in modules:
        try:
            __import__(name)
            print(f"ok   {name}")
        except Exception as exc:  # noqa: BLE001 — surface any import failure
            print(f"FAIL {name}: {exc}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} import(s) failed. Use repo venv: python -m venv venv && pip install -r requirements.txt")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

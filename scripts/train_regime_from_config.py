#!/usr/bin/env python3
"""Train a regime detector from YAML (wrapper for ``phi.regime.cli_train``)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from phi.regime.cli_train import main as _main

    _main()


if __name__ == "__main__":
    main()

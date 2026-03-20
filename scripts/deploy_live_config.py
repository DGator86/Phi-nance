#!/usr/bin/env python3
"""Deploy an optimized config JSON as the active live config."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from phi.config import settings
from phi.live.loader import resolve_latest_best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--best-dir", type=Path, default=Path("runs/best_params"))
    args = parser.parse_args()

    source = args.source or resolve_latest_best_params(args.best_dir)
    if source is None:
        raise SystemExit("No source config found")
    settings.LIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, settings.LIVE_CONFIG_PATH)
    print(f"Deployed {source} -> {settings.LIVE_CONFIG_PATH}")


if __name__ == "__main__":
    main()

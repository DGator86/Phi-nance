#!/usr/bin/env python3
"""CLI entry point for the phi live trader."""

from __future__ import annotations

import argparse
from pathlib import Path

from phi.live.engine import LiveEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start live/paper trading engine")
    parser.add_argument("--config", type=Path, default=None, help="Path to live config JSON")
    parser.add_argument("--cycles", type=int, default=None, help="Optional max cycles for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = LiveEngine.from_settings(config_path=args.config)
    engine.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()

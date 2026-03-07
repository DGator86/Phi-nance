"""CLI entrypoint for experiment sweeps."""

from __future__ import annotations

import argparse
import json
import logging

from phinance.experiment.sweep import SweepRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Phi-nance hyperparameter sweep")
    parser.add_argument("--config", required=True, help="Path to sweep YAML config")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = SweepRunner(args.config).run()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

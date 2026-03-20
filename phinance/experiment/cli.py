"""CLI entrypoints for experiment execution."""

from __future__ import annotations

import argparse
import json
import logging

from phinance.experiment.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Phi-nance experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with dotted keys, e.g. params.timesteps=1000",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_experiment(args.config, overrides=args.override)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train hierarchical meta-agent for option coordination."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from phinance.rl.hierarchical.training import load_config, train_with_areal, train_with_fallback_loop
from phinance.rl.training_utils import load_optimisation_config

logger = logging.getLogger(__name__)


def train_meta_agent(
    config: str = "configs/meta_agent.yaml",
    output: str = "models/meta_agent",
    optim_config: str = "configs/rl_optimisation_config.yaml",
    fallback: bool = True,
    tracker: Any = None,
) -> dict[str, float]:
    config_path = Path(config)
    output_path = Path(output)
    optim_config_path = Path(optim_config)

    cfg = load_config(config_path)
    optim_cfg = load_optimisation_config(optim_config_path)
    if tracker is not None:
        tracker.log_params(
            {
                "config": str(config_path),
                "optim_config": str(optim_config_path),
                "fallback": bool(fallback),
            }
        )

    if fallback:
        checkpoint, metrics = train_with_fallback_loop(cfg, output_path, optim_cfg, tracker=tracker)
    else:
        checkpoint, metrics = train_with_areal(cfg, output_path, optim_cfg, tracker=tracker)

    if tracker is not None:
        tracker.log_artifact(str(checkpoint))

    metrics["checkpoint_size_bytes"] = float(checkpoint.stat().st_size if checkpoint.exists() else 0)
    metrics["used_fallback"] = float(bool(fallback))
    return metrics


def run_experiment_target(
    config: str = "configs/meta_agent.yaml",
    output: str = "models/meta_agent",
    optim_config: str = "configs/rl_optimisation_config.yaml",
    fallback: bool = True,
    tracker: Any = None,
) -> dict[str, float]:
    return train_meta_agent(config, output, optim_config, fallback, tracker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical meta-agent")
    parser.add_argument("--config", type=Path, default=Path("configs/meta_agent.yaml"))
    parser.add_argument("--output", type=Path, default=Path("models/meta_agent"))
    parser.add_argument("--optim-config", type=Path, default=Path("configs/rl_optimisation_config.yaml"))
    parser.add_argument("--fallback", action="store_true", help="Use fallback random smoke loop")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    optim_cfg = load_optimisation_config(args.optim_config)

    if args.fallback:
        checkpoint, _ = train_with_fallback_loop(config, args.output, optim_cfg)
    else:
        checkpoint, _ = train_with_areal(config, args.output, optim_cfg)

    logger.info("Saved meta-agent checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

"""Train hierarchical meta-agent for option coordination."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from phinance.rl.hierarchical.training import load_config, train_with_areal, train_with_fallback_loop
from phinance.rl.training_utils import load_optimisation_config

logger = logging.getLogger(__name__)


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
        checkpoint = train_with_fallback_loop(config, args.output, optim_cfg)
    else:
        checkpoint = train_with_areal(config, args.output, optim_cfg)

    logger.info("Saved meta-agent checkpoint to %s", checkpoint)


if __name__ == "__main__":
    main()

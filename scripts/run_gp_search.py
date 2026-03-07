"""Run GP strategy discovery as a trackable experiment target."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phinance.meta.search import run_meta_search


def _synthetic_ohlcv(rows: int = 250, seed: int = 21) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.0, size=rows))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(900, 1800, rows),
        },
        index=idx,
    )


def run_gp_search(
    config_path: str = "configs/meta_config.yaml",
    vault_path: str = "data/strategy_vault.json",
    distributed_config_path: str = "configs/distributed_config.yaml",
    rows: int = 250,
    seed: int = 21,
    tracker: Any = None,
) -> dict[str, float]:
    if tracker is not None:
        tracker.log_params(
            {
                "config_path": config_path,
                "vault_path": vault_path,
                "distributed_config_path": distributed_config_path,
                "rows": rows,
                "seed": seed,
            }
        )

    data = _synthetic_ohlcv(rows=rows, seed=seed)
    result = run_meta_search(
        data,
        config_path=config_path,
        vault_path=vault_path,
        distributed_config_path=distributed_config_path,
    )

    best = result.get("best_strategy") or {}
    metrics = {
        "best_fitness": float(best.get("fitness", 0.0)),
        "strategy_count": float(len(result.get("best_strategies", []))),
    }
    if tracker is not None:
        tracker.log_metrics(metrics)
        if Path(vault_path).exists():
            tracker.log_artifact(vault_path)

    return metrics


def run_experiment_target(
    config_path: str = "configs/meta_config.yaml",
    vault_path: str = "data/strategy_vault.json",
    distributed_config_path: str = "configs/distributed_config.yaml",
    rows: int = 250,
    seed: int = 21,
    tracker: Any = None,
) -> dict[str, float]:
    return run_gp_search(config_path, vault_path, distributed_config_path, rows, seed, tracker)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GP strategy discovery")
    parser.add_argument("--config-path", default="configs/meta_config.yaml")
    parser.add_argument("--vault-path", default="data/strategy_vault.json")
    parser.add_argument("--distributed-config-path", default="configs/distributed_config.yaml")
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    metrics = run_gp_search(
        config_path=args.config_path,
        vault_path=args.vault_path,
        distributed_config_path=args.distributed_config_path,
        rows=args.rows,
        seed=args.seed,
    )
    print(metrics)


if __name__ == "__main__":
    main()

"""Top-level search runner for meta-learning strategy discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from phinance.meta.genetic import GPConfig, GeneticStrategySearch
from phinance.meta.vault_integration import save_to_vault


def load_meta_config(path: str | Path = "configs/meta_config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _load_distributed_config(path: str | Path = "configs/distributed_config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    full = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return full.get("distributed", {})


def run_meta_search(
    ohlcv: pd.DataFrame,
    config_path: str | Path = "configs/meta_config.yaml",
    vault_path: str | Path = "data/strategy_vault.json",
    distributed_config_path: str | Path = "configs/distributed_config.yaml",
) -> Dict[str, Any]:
    cfg = load_meta_config(config_path)
    gp_overrides = dict(cfg.get("gp", {}))

    distributed = _load_distributed_config(distributed_config_path)
    gp_overrides.setdefault("distributed_enabled", bool(distributed.get("enabled", False)))
    gp_overrides.setdefault("distributed_use_ray", bool(distributed.get("use_ray", True)))
    gp_overrides.setdefault("distributed_num_cpus", distributed.get("num_cpus"))
    gp_overrides.setdefault("distributed_address", distributed.get("address"))
    gp_overrides.setdefault("distributed_timeout_s", distributed.get("timeout_s"))

    gp_cfg = GPConfig(**gp_overrides)

    runner = GeneticStrategySearch(ohlcv=ohlcv, config=gp_cfg)
    result = runner.evolve()
    save_to_vault(result["best_strategies"], path=vault_path)
    return result

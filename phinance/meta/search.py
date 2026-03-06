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


def run_meta_search(
    ohlcv: pd.DataFrame,
    config_path: str | Path = "configs/meta_config.yaml",
    vault_path: str | Path = "data/strategy_vault.json",
) -> Dict[str, Any]:
    cfg = load_meta_config(config_path)
    gp_cfg = GPConfig(**cfg.get("gp", {}))

    runner = GeneticStrategySearch(ohlcv=ohlcv, config=gp_cfg)
    result = runner.evolve()
    save_to_vault(result["best_strategies"], path=vault_path)
    return result

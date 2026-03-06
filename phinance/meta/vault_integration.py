"""Strategy vault helpers for discovered GP strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_VAULT_PATH = Path("data/strategy_vault.json")


def load_vault(path: str | Path = DEFAULT_VAULT_PATH) -> Dict[str, Any]:
    vault_path = Path(path)
    if not vault_path.exists():
        return {"strategies": []}
    return json.loads(vault_path.read_text(encoding="utf-8"))


def save_to_vault(strategies: List[Dict[str, Any]], path: str | Path = DEFAULT_VAULT_PATH) -> Path:
    vault_path = Path(path)
    vault_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_vault(vault_path)
    existing = payload.get("strategies", [])
    payload["strategies"] = existing + strategies
    vault_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return vault_path


def load_discovered_templates(path: str | Path = DEFAULT_VAULT_PATH) -> List[Dict[str, Any]]:
    payload = load_vault(path)
    templates: List[Dict[str, Any]] = []
    for strat in payload.get("strategies", []):
        templates.append(
            {
                "name": "gp_discovered_strategy",
                "params": {
                    "strategy_id": strat.get("strategy_id"),
                    "expression": strat.get("expression"),
                    "source": "meta_gp",
                },
            }
        )
    return templates

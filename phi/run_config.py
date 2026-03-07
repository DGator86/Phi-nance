"""Run configuration schema and run-history persistence helpers."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_ROOT = _PROJECT_ROOT / "runs"


def _ensure_runs_dir() -> Path:
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return _RUNS_ROOT


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


class RunConfig(BaseModel):
    """Validated, versioned backtest run configuration."""

    model_config = ConfigDict(extra="allow")

    dataset_id: str = ""
    symbols: List[str] = Field(default_factory=lambda: ["SPY"])
    start_date: date
    end_date: date
    timeframe: str = "1D"
    vendor: str = "alphavantage"
    initial_capital: float = 100_000.0
    trading_mode: Literal["equities", "options"] = "equities"
    indicators: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    blend_method: str = "weighted_sum"
    blend_weights: Dict[str, float] = Field(default_factory=dict)
    phiai_enabled: bool = False
    phiai_constraints: Dict[str, Any] = Field(default_factory=dict)
    exit_rules: Dict[str, Any] = Field(default_factory=dict)
    position_sizing: Dict[str, Any] = Field(default_factory=dict)
    evaluation_metric: str = "roi"
    schema_version: int = Field(default=1, frozen=True)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: List[str]) -> List[str]:
        symbols = [s.strip().upper() for s in value if str(s).strip()]
        if not symbols:
            raise ValueError("symbols must contain at least one symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        return symbols

    @field_validator("initial_capital")
    @classmethod
    def validate_initial_capital(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("initial_capital must be > 0")
        return value

    @field_validator("indicators")
    @classmethod
    def validate_indicators(cls, value: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        validated: Dict[str, Dict[str, Any]] = {}
        for name, payload in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("indicator names must be non-empty strings")
            if not isinstance(payload, dict):
                raise ValueError(f"indicator '{name}' must map to a dict")
            if "enabled" not in payload or not isinstance(payload["enabled"], bool):
                raise ValueError(f"indicator '{name}' must include enabled: bool")
            if "params" in payload and not isinstance(payload["params"], dict):
                raise ValueError(f"indicator '{name}' params must be a dict")
            validated[name] = payload
        return validated

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> "RunConfig":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be less than or equal to end_date")

        today = date.today()
        if self.start_date > today or self.end_date > today:
            logger.warning(
                "RunConfig uses future date(s): start_date=%s end_date=%s",
                self.start_date,
                self.end_date,
            )

        if self.blend_method == "weighted_sum":
            enabled = {
                name
                for name, cfg in self.indicators.items()
                if isinstance(cfg, dict) and cfg.get("enabled") is True
            }
            if enabled and not self.blend_weights:
                raise ValueError(
                    "blend_weights are required when blend_method='weighted_sum' and indicators are enabled"
                )
            if self.blend_weights:
                keys = set(self.blend_weights.keys())
                if keys != enabled:
                    raise ValueError(
                        "blend_weights keys must match enabled indicators. "
                        f"enabled={sorted(enabled)}, weights={sorted(keys)}"
                    )
                total = sum(float(v) for v in self.blend_weights.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError("blend_weights values must sum to 1.0")
        return self

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return a serializable dict including schema_version."""
        kwargs.setdefault("mode", "json")
        payload = super().model_dump(*args, **kwargs)
        payload["schema_version"] = 1
        return payload

    def to_dict(self) -> Dict[str, Any]:
        """Compatibility helper for legacy callers."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        """Create a RunConfig from a dict payload."""
        return cls.model_validate(d)

    @classmethod
    def _migrate_v0_to_v1(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        migrated = dict(payload)
        if "start" in migrated and "start_date" not in migrated:
            start = migrated.pop("start")
            migrated["start_date"] = start.date().isoformat() if hasattr(start, "date") else str(start)
        if "end" in migrated and "end_date" not in migrated:
            end = migrated.pop("end")
            migrated["end_date"] = end.date().isoformat() if hasattr(end, "date") else str(end)
        migrated.setdefault("schema_version", 1)
        return migrated

    @classmethod
    def from_json(cls, filepath: Path) -> "RunConfig":
        """Load, migrate (if needed), and validate config JSON from disk."""
        with open(filepath, encoding="utf-8") as f:
            raw = json.load(f)

        version = int(raw.get("schema_version", 0))
        payload = raw
        if version < 1:
            payload = cls._migrate_v0_to_v1(raw)

        try:
            cfg = cls.model_validate(payload)
            logger.info("Loaded RunConfig from %s (schema v%s)", filepath, cfg.schema_version)
            return cfg
        except ValidationError:
            logger.warning("RunConfig validation failed for %s", filepath, exc_info=True)
            raise

    @classmethod
    def load(cls, run_dir: Path) -> "RunConfig":
        """Load config.json from a run directory."""
        return cls.from_json(run_dir / "config.json")

    def save(self, run_dir: Path) -> None:
        """Save this config to run_dir/config.json."""
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


class RunHistory:
    """Manages run storage and retrieval."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _RUNS_ROOT

    def create_run(self, config: RunConfig) -> str:
        run_id = _new_run_id()
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        config.save(path)
        return run_id

    def save_results(self, run_id: str, results: Dict[str, Any], trades: Optional[pd.DataFrame] = None) -> None:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        if trades is not None and not trades.empty:
            trades.to_csv(path / "trades.csv", index=False)

    def load_config(self, run_id: str) -> Optional[RunConfig]:
        path = self.root / run_id / "config.json"
        if not path.exists():
            return None
        return RunConfig.from_json(path)

    def load_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.root / run_id / "results.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_trades(self, run_id: str) -> Optional[pd.DataFrame]:
        path = self.root / run_id / "trades.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with basic metadata."""
        runs = []
        if not self.root.exists():
            return runs
        for d in sorted(self.root.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            cfg = self.load_config(d.name)
            res = self.load_results(d.name)
            runs.append({
                "run_id": d.name,
                "config": cfg.model_dump() if cfg else {},
                "results": res or {},
            })
        return runs[:100]

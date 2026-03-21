"""Run configuration schema and run-history persistence helpers."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from phi.config import settings
from phi.exceptions import ValidationError as PhiValidationError
from phi.logging import get_logger
from phi.utils.validation import resolve_safe_path, sanitize_run_id

logger = get_logger(__name__)

_RUNS_ROOT = settings.RUNS_DIR


def _ensure_runs_dir() -> Path:
    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return _RUNS_ROOT


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


class RunConfig(BaseModel):
    """Validated, versioned backtest run configuration."""

    model_config = ConfigDict(extra="allow")

    dataset_id: str = ""
    symbols: list[str] = Field(default_factory=lambda: ["SPY"])
    start_date: date
    end_date: date
    timeframe: str = "1D"
    vendor: str = "unusual_whales"
    initial_capital: float = 100_000.0
    trading_mode: Literal["equities", "options"] = "equities"
    option_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    indicators: dict[str, dict[str, Any]] = Field(default_factory=dict)
    blend_method: str = "weighted_sum"
    blend_weights: dict[str, float] = Field(default_factory=dict)
    phiai_enabled: bool = False
    phiai_n_trials: int | None = Field(default=None, ge=1)
    phiai_walk_forward_windows: int | None = Field(default=None, ge=1)
    phiai_parallel_jobs: int | None = Field(default=None, ge=1)
    phiai_constraints: dict[str, Any] = Field(default_factory=dict)
    regime_detector_params: dict[str, Any] | None = None
    regime_boosts: dict[str, dict[str, float]] | None = None
    exit_rules: dict[str, Any] = Field(default_factory=dict)
    position_sizing: dict[str, Any] = Field(default_factory=dict)
    evaluation_metric: str = "roi"
    allocation_strategy: str = "equal_weight"
    allocation_params: dict[str, Any] = Field(default_factory=dict)
    rebalance_frequency: str | int | None = "M"
    rebalance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # Optional inline override for options regime playbook (see phi.options.regime_playbook)
    options_regime_playbook: dict[str, Any] | None = None
    schema_version: int = Field(default=1, frozen=True)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, value: list[str]) -> list[str]:
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
    def validate_indicators(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        validated: dict[str, dict[str, Any]] = {}
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


    @field_validator("allocation_strategy")
    @classmethod
    def validate_allocation_strategy(cls, value: str) -> str:
        key = str(value).strip().lower()
        allowed = {"equal_weight", "equal", "fixed_weight", "fixed", "signal_weighted", "signal", "risk_parity"}
        if key not in allowed:
            raise ValueError(f"allocation_strategy must be one of {sorted(allowed)}")
        return key

    @model_validator(mode="after")
    def validate_cross_field_rules(self) -> RunConfig:
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
                    logger.warning("blend_weights sum is %.6f (expected 1.0)", total)
                    raise ValueError("blend_weights values must sum to 1.0")

        if (self.regime_detector_params is None) != (self.regime_boosts is None):
            raise ValueError("regime_detector_params and regime_boosts must both be provided together")
        if self.regime_detector_params is not None and self.regime_boosts is not None:
            n_regimes = int(self.regime_detector_params.get("n_regimes", 0))
            if n_regimes <= 0:
                raise ValueError("regime_detector_params.n_regimes must be > 0 when regime settings are provided")
            expected = {str(i) for i in range(n_regimes)}
            got = {str(k) for k in self.regime_boosts.keys()}
            if got != expected:
                raise ValueError(
                    "regime_boosts keys must match detector n_regimes. "
                    f"expected={sorted(expected)}, got={sorted(got)}"
                )

        if self.allocation_strategy in {"fixed_weight", "fixed"}:
            weights = self.allocation_params.get("weights", {}) if isinstance(self.allocation_params, dict) else {}
            if not isinstance(weights, dict) or not weights:
                raise ValueError("allocation_params.weights is required for fixed allocation")
            total = sum(float(v) for v in weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError("allocation_params.weights must sum to 1.0")

        if self.trading_mode == "options":
            if len(self.symbols) != 1:
                raise ValueError("options mode currently supports exactly one symbol")
            symbol = self.symbols[0]
            params = self.option_params.get(symbol)
            if not isinstance(params, dict):
                raise ValueError(f"option_params must contain configuration for symbol '{symbol}'")
            required = {"option_type", "strike", "expiry"}
            missing = sorted(k for k in required if k not in params)
            if missing:
                raise ValueError(f"option_params['{symbol}'] missing required keys: {missing}")
        return self

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a serializable dict including schema_version."""
        kwargs.setdefault("mode", "json")
        payload = super().model_dump(*args, **kwargs)
        payload["schema_version"] = 1
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Compatibility helper for legacy callers."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunConfig:
        """Create a RunConfig from a dict payload."""
        return cls.model_validate(d)

    @classmethod
    def _migrate_v0_to_v1(cls, payload: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(payload)
        if "start" in migrated and "start_date" not in migrated:
            start = migrated.pop("start")
            migrated["start_date"] = start.date().isoformat() if hasattr(start, "date") else str(start)
        if "end" in migrated and "end_date" not in migrated:
            end = migrated.pop("end")
            migrated["end_date"] = end.date().isoformat() if hasattr(end, "date") else str(end)
        migrated.setdefault("schema_version", 1)
        logger.info("Migrated legacy RunConfig payload from v0 to v1")
        return migrated

    @classmethod
    def from_json(cls, filepath: Path) -> RunConfig:
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
        except PydanticValidationError as exc:
            logger.warning("RunConfig validation failed for %s: %s", filepath, exc, exc_info=True)
            raise

    @classmethod
    def load(cls, run_dir: Path) -> RunConfig:
        """Load config.json from a run directory."""
        return cls.from_json(run_dir / "config.json")

    def save(self, run_dir: Path) -> None:
        """Save this config to run_dir/config.json."""
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)


class RunHistory:
    """Manages run storage and retrieval."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or settings.RUNS_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, config: RunConfig) -> str:
        run_id = _new_run_id()
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        config.save(path)
        return run_id

    def save_results(self, run_id: str, results: dict[str, Any], trades: pd.DataFrame | None = None) -> None:
        safe_run_id = sanitize_run_id(run_id)
        path = resolve_safe_path(self.root, safe_run_id)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        if trades is not None and not trades.empty:
            trades.to_csv(path / "trades.csv", index=False)

    def load_config(self, run_id: str) -> RunConfig | None:
        try:
            path = resolve_safe_path(self.root, run_id) / "config.json"
        except PhiValidationError:
            return None
        if not path.exists():
            return None
        return RunConfig.from_json(path)

    def load_results(self, run_id: str) -> dict[str, Any] | None:
        try:
            path = resolve_safe_path(self.root, run_id) / "results.json"
        except PhiValidationError:
            return None
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def load_trades(self, run_id: str) -> pd.DataFrame | None:
        try:
            path = resolve_safe_path(self.root, run_id) / "trades.csv"
        except PhiValidationError:
            return None
        if not path.exists():
            return None
        return pd.read_csv(path)

    def list_runs(self) -> list[dict[str, Any]]:
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

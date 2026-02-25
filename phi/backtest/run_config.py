"""
phi.backtest.run_config — RunConfig Schema
==========================================
Captures every parameter needed to reproduce a backtest run.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class IndicatorConfig:
    name:        str
    display_name: str
    params:      Dict[str, Any] = field(default_factory=dict)
    auto_tuned:  bool = False
    enabled:     bool = True
    weight:      float = 1.0


@dataclass
class ExitRules:
    stop_loss_pct:     Optional[float] = None   # e.g. 0.02 = 2%
    take_profit_pct:   Optional[float] = None   # e.g. 0.05 = 5%
    trailing_stop_pct: Optional[float] = None   # e.g. 0.02 = 2%
    time_exit_bars:    Optional[int]   = None   # exit after N bars
    signal_exit:       bool = True              # exit on opposite signal


@dataclass
class PositionSizing:
    method:       str   = "fixed_pct"           # "fixed_pct" | "fixed_shares"
    pct_of_cash:  float = 0.95
    fixed_shares: int   = 0
    allow_short:  bool  = False


@dataclass
class OptionsConfig:
    structure:      str   = "long_call"          # options structure type
    expiry_rule:    str   = "nearest"            # "nearest" | "fixed_dte" | "delta"
    target_dte:     int   = 45
    profit_exit_pct: float = 0.50               # exit at 50% profit
    stop_exit_pct:   float = 1.00               # exit at 100% loss (full debit)


@dataclass
class RunConfig:
    # Identifiers
    run_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at:   str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Dataset
    dataset_id:   str  = ""
    symbol:       str  = "SPY"
    timeframe:    str  = "1D"
    start_date:   str  = ""
    end_date:     str  = ""
    vendor:       str  = "yfinance"

    # Capital
    initial_capital: float = 100_000.0

    # Trading mode
    trading_mode: str  = "equities"             # "equities" | "options"

    # Indicators
    indicators:   List[IndicatorConfig] = field(default_factory=list)

    # Blending
    blend_mode:   str  = "weighted_sum"
    blend_weights: Dict[str, float] = field(default_factory=dict)
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # PhiAI
    phiai_enabled:   bool = False
    phiai_max_inds:  int  = 3
    phiai_no_short:  bool = True
    phiai_risk_cap:  float = 0.02

    # Exit rules
    exit_rules: ExitRules = field(default_factory=ExitRules)

    # Position sizing
    position_sizing: PositionSizing = field(default_factory=PositionSizing)

    # Options config (if trading_mode == "options")
    options_config: OptionsConfig = field(default_factory=OptionsConfig)

    # Evaluation
    evaluation_metric: str = "sharpe"           # roi | cagr | sharpe | max_dd | accuracy | profit_factor

    # Signal parameters
    signal_threshold: float = 0.10             # minimum |signal| to act on

    # Metadata / notes
    description:  str = ""
    tags:         List[str] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        # Reconstruct nested dataclasses
        inds = [IndicatorConfig(**i) for i in d.pop("indicators", [])]
        er   = ExitRules(**d.pop("exit_rules", {}))
        ps   = PositionSizing(**d.pop("position_sizing", {}))
        oc   = OptionsConfig(**d.pop("options_config", {}))
        obj  = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.indicators     = inds
        obj.exit_rules     = er
        obj.position_sizing = ps
        obj.options_config  = oc
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        return cls.from_dict(json.loads(s))

    def fingerprint(self) -> str:
        """Stable hash for deduplication (excludes run_id, created_at)."""
        d = self.to_dict()
        for k in ("run_id", "created_at"):
            d.pop(k, None)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

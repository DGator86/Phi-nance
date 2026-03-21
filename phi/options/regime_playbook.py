"""Versioned options regime playbooks: structures, risk bands, and transition map.

Bridges detailed regime labels (``BULL_LOW_VOL``, …) from ``REGIME_STRATEGY_MAP``
with human-readable guidance for the workbench and external execution adapters.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from phi.logging import get_logger
from phi.regime.regime_definitions import (
    compose_detailed_regime,
    infer_base_regime_from_prices,
    infer_volatility_regime,
)
from phi.regime.strategy_mapping import REGIME_STRATEGY_MAP

logger = get_logger(__name__)

_DEFAULT_TRANSITIONS: list[dict[str, str]] = [
    {
        "from": "RANGING_LOW_VOL",
        "to": "RANGING_HIGH_VOL",
        "trigger": "Realised vol jumps into upper quantile; IV lift or whale flow spike",
        "action": "Favor long vol (straddle/strangle); cut naked short premium size",
    },
    {
        "from": "BULL_NORMAL_VOL",
        "to": "BEAR_HIGH_VOL",
        "trigger": "Trend break + vol expansion (risk-off)",
        "action": "Reduce directional long delta; prefer defined-risk bearish structures",
    },
    {
        "from": "RANGING_HIGH_VOL",
        "to": "RANGING_LOW_VOL",
        "trigger": "Vol crush after event; range re-establishes",
        "action": "Iron condor / calendar reasonable again; tighten event risk",
    },
    {
        "from": "BULL_LOW_VOL",
        "to": "BULL_HIGH_VOL",
        "trigger": "Trend intact but vol regime shifts up",
        "action": "Widen spreads or add upside convexity; avoid tight short calls",
    },
]


class PlaybookTransition(BaseModel):
    """One row in the regime prediction / response map."""

    model_config = ConfigDict(populate_by_name=True)

    from_regime: str = Field(validation_alias="from", serialization_alias="from")
    to_regime: str = Field(validation_alias="to", serialization_alias="to")
    trigger: str = ""
    action: str = ""


class RegimePlaybookEntry(BaseModel):
    """Per-regime options guidance (structures + risk envelope)."""

    model_config = ConfigDict(extra="ignore")

    display_name: str
    summary: str = ""
    allowed_structures: list[str] = Field(default_factory=list)
    dte_days_min: int = 7
    dte_days_max: int = 120
    delta_band: tuple[float, float] = (0.30, 0.55)
    max_risk_pct_portfolio: float = Field(default=0.05, ge=0.0, le=1.0)
    notes: str = ""


class OptionsRegimePlaybook(BaseModel):
    """Full playbook document (JSON-serializable, versioned)."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0"
    regimes: dict[str, RegimePlaybookEntry]
    transitions: list[PlaybookTransition] = Field(default_factory=list)


def _entry_for_regime_key(key: str, strategies: tuple[str, ...]) -> RegimePlaybookEntry:
    k = key.upper()
    pretty = k.replace("_", " ").title()
    return RegimePlaybookEntry(
        display_name=pretty,
        summary=f"Preferred structures for {pretty} (trend × vol composite).",
        allowed_structures=list(strategies),
        dte_days_min=14,
        dte_days_max=90,
        delta_band=(0.35, 0.55),
        max_risk_pct_portfolio=0.04,
        notes="Whales, flow, and news should update your belief over these labels — not replace them.",
    )


def build_default_options_regime_playbook() -> OptionsRegimePlaybook:
    """Derive playbook rows from ``REGIME_STRATEGY_MAP`` plus default transition hints."""
    regimes = {
        key: _entry_for_regime_key(key, tpl) for key, tpl in REGIME_STRATEGY_MAP.items()
    }
    transitions = [PlaybookTransition.model_validate(t) for t in _DEFAULT_TRANSITIONS]
    return OptionsRegimePlaybook(version="1.0", regimes=regimes, transitions=transitions)


@lru_cache(maxsize=1)
def get_default_options_regime_playbook() -> OptionsRegimePlaybook:
    """Cached default playbook (process lifetime)."""
    return build_default_options_regime_playbook()


def load_options_regime_playbook(path: Path | None = None) -> OptionsRegimePlaybook:
    """Load playbook from JSON path, env ``PHINANCE_OPTIONS_PLAYBOOK``, or built-in default."""
    raw_path = path or os.environ.get("PHINANCE_OPTIONS_PLAYBOOK", "").strip()
    if raw_path:
        p = Path(raw_path).expanduser()
        if p.is_file():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return OptionsRegimePlaybook.model_validate(data)
        logger.warning("Playbook path not found: %s — using default", p)
    return get_default_options_regime_playbook()


def resolve_playbook_regime_key(raw: str, playbook: OptionsRegimePlaybook | None = None) -> str | None:
    """Map arbitrary label (e.g. detector state) to a playbook regime key if possible."""
    pb = playbook or get_default_options_regime_playbook()
    s = str(raw).strip().upper().replace(" ", "_")
    if s in pb.regimes:
        return s
    for key in pb.regimes:
        if key.upper() == s:
            return key
    return None


def quick_detailed_regime_from_ohlcv(ohlcv: pd.DataFrame) -> str:
    """Infer composite regime label from price using trend + realised vol quantiles."""
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 30:
        return "RANGING_NORMAL_VOL"
    base = infer_base_regime_from_prices(ohlcv)
    vol = infer_volatility_regime(ohlcv)
    return compose_detailed_regime(base, vol).label


def playbook_entry_for_label(label: str, playbook: OptionsRegimePlaybook | None = None) -> RegimePlaybookEntry | None:
    """Return playbook entry for a detailed regime label, or None."""
    pb = playbook or get_default_options_regime_playbook()
    key = resolve_playbook_regime_key(label, pb)
    if key is None:
        return None
    return pb.regimes.get(key)


def playbook_to_summary_dict(playbook: OptionsRegimePlaybook) -> dict[str, Any]:
    """Compact dict for Streamlit / API (JSON-friendly)."""
    return {
        "version": playbook.version,
        "regime_keys": list(playbook.regimes.keys()),
        "transitions": [t.model_dump(mode="json", by_alias=True) for t in playbook.transitions],
    }

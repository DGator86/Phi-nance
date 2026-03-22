"""Structured options trade suggestion: entry, targets, MTF, regime, info metrics, ML hook."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OptionsSignalCard(BaseModel):
    """Single actionable options context card (not a guaranteed fill price)."""

    model_config = {"extra": "ignore"}

    symbol: str
    composite_regime: str = ""
    playbook_regime_key: str | None = None

    # Data lineage (Unusual Whales → yfinance via ``phi.data.fetch_ohlcv_uw_then_yf``)
    ohlcv_vendor: str = ""
    unusual_whales_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="ATM chain row + flow summary when UW API key is set and enrich succeeds.",
    )

    action: Literal["ENTER", "WAIT", "SKIP"] = "WAIT"
    structure: str = ""
    structure_rationale: str = ""

    # Entry / risk envelope (aligns with phi.options.simulator / engine defaults)
    entry_trigger: str = ""
    target_exit_pct: float = Field(default=0.50, description="Take-profit vs premium / allocated (e.g. 0.5 = +50%)")
    stop_exit_pct: float = Field(default=1.00, description="Stop vs premium / allocated (e.g. 1.0 = -100%)")
    dte_days_min: int = 14
    dte_days_max: int = 90
    delta_band_low: float = 0.35
    delta_band_high: float = 0.55
    max_risk_pct_portfolio: float = 0.04

    # MTF
    mtf_timeframes_present: list[str] = Field(default_factory=list)
    mtf_confluence: float | None = None
    mtf_alignment: Literal["WITH_TREND", "AGAINST", "MIXED", "N_A"] = "N_A"
    mtf_notes: str = ""

    # Information theory (last bar), normalized signals where applicable
    info_metrics: dict[str, float] = Field(default_factory=dict)
    info_notes: str = ""

    # Regime reasoning (deterministic bullets)
    reasoning: list[str] = Field(default_factory=list)

    # PhiAI / promoted params
    phiai_dataset_id: str | None = None
    phiai_promoted: bool = False
    phiai_metric: str | None = None
    phiai_best_value: float | None = None

    def to_display_dict(self) -> dict[str, Any]:
        """Flat dict for Streamlit / JSON."""
        return self.model_dump()

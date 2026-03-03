"""
phinance.config.schema
=======================

Validation schemas and helpers for RunConfig and related structures.

Functions
---------
  validate_run_config(cfg)     — Raise ConfigurationError on invalid config
  validate_indicators(ind)     — Check indicator dict structure
  validate_blend(method, w)    — Check blend method + weights dict
  validate_date_range(s, e)    — Raise if end < start
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from phinance.exceptions import ConfigurationError
from phinance.blending.methods import BLEND_METHODS


def validate_run_config(cfg: "phinance.config.run_config.RunConfig") -> None:  # type: ignore[name-defined]
    """Run all validation checks on a RunConfig instance.

    Raises
    ------
    ConfigurationError
    """
    cfg.validate()  # Delegate to RunConfig.validate()
    if cfg.start_date and cfg.end_date:
        validate_date_range(cfg.start_date, cfg.end_date)
    if cfg.indicators:
        validate_indicators(cfg.indicators)
    validate_blend(cfg.blend_method, cfg.blend_weights)


def validate_indicators(indicators: Dict[str, Any]) -> None:
    """Assert that an indicators dict has the expected shape.

    Each value must be a dict with at least an ``"enabled"`` key.

    Raises
    ------
    ConfigurationError
    """
    if not isinstance(indicators, dict):
        raise ConfigurationError(
            f"'indicators' must be a dict; got {type(indicators).__name__}"
        )
    for name, cfg in indicators.items():
        if not isinstance(cfg, dict):
            raise ConfigurationError(
                f"Indicator '{name}' config must be a dict; got {type(cfg).__name__}"
            )


def validate_blend(method: str, weights: Optional[Dict[str, float]]) -> None:
    """Assert that blend method is valid and weights are non-negative.

    Raises
    ------
    ConfigurationError
    """
    if method not in BLEND_METHODS:
        raise ConfigurationError(
            f"Invalid blend_method: '{method}'. Supported: {BLEND_METHODS}"
        )
    if weights:
        for name, w in weights.items():
            if float(w) < 0:
                raise ConfigurationError(
                    f"Blend weight for '{name}' must be non-negative; got {w}"
                )


def validate_date_range(start: str, end: str) -> None:
    """Raise ConfigurationError if end < start.

    Parameters
    ----------
    start, end : str — ``"YYYY-MM-DD"``
    """
    import pandas as pd

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if e < s:
        raise ConfigurationError(
            f"end_date ({end}) must be >= start_date ({start})"
        )

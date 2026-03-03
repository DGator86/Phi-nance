"""
phinance.config.run_config
===========================

RunConfig — reproducible backtest configuration schema.

A RunConfig captures every parameter needed to reproduce a backtest run:
symbols, dates, vendor, indicators, blend settings, PhiAI flags, etc.

It is intentionally plain Python (no Pydantic dependency) so the library
remains lightweight; validation is done via the ``validate()`` method.

Usage
-----
    from phinance.config.run_config import RunConfig

    cfg = RunConfig(
        symbols    = ["SPY"],
        start_date = "2022-01-01",
        end_date   = "2023-12-31",
        timeframe  = "1D",
    )
    d = cfg.to_dict()
    cfg2 = RunConfig.from_dict(d)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RunConfig:
    """Reproducible backtest configuration.

    Parameters
    ----------
    dataset_id        : str — optional dataset label
    symbols           : list[str] — ticker symbols (default: ["SPY"])
    start_date        : str — ``"YYYY-MM-DD"``
    end_date          : str — ``"YYYY-MM-DD"``
    timeframe         : str — ``"1D"`` | ``"1H"`` | ``"15m"`` | ...
    vendor            : str — data vendor key
    initial_capital   : float
    trading_mode      : ``"equities"`` | ``"options"``
    indicators        : dict — ``{name: {"enabled": bool, "params": {...}}}``
    blend_method      : str
    blend_weights     : dict — ``{name: weight}``
    phiai_enabled     : bool
    phiai_constraints : dict
    exit_rules        : dict
    position_sizing   : dict
    evaluation_metric : str — e.g. ``"roi"`` | ``"sharpe"``
    """

    def __init__(
        self,
        dataset_id:        str = "",
        symbols:           Optional[List[str]] = None,
        start_date:        str = "",
        end_date:          str = "",
        timeframe:         str = "1D",
        vendor:            str = "alphavantage",
        initial_capital:   float = 100_000.0,
        trading_mode:      str = "equities",
        indicators:        Optional[Dict[str, Dict[str, Any]]] = None,
        blend_method:      str = "weighted_sum",
        blend_weights:     Optional[Dict[str, float]] = None,
        phiai_enabled:     bool = False,
        phiai_constraints: Optional[Dict[str, Any]] = None,
        exit_rules:        Optional[Dict[str, Any]] = None,
        position_sizing:   Optional[Dict[str, Any]] = None,
        evaluation_metric: str = "roi",
        **extra: Any,
    ) -> None:
        self.dataset_id        = dataset_id
        # Allow explicit empty list (for validation testing); None defaults to ["SPY"]
        self.symbols           = symbols if symbols is not None else ["SPY"]
        self.start_date        = start_date
        self.end_date          = end_date
        self.timeframe         = timeframe
        self.vendor            = vendor
        self.initial_capital   = initial_capital
        self.trading_mode      = trading_mode
        self.indicators        = indicators or {}
        self.blend_method      = blend_method
        self.blend_weights     = blend_weights or {}
        self.phiai_enabled     = phiai_enabled
        self.phiai_constraints = phiai_constraints or {}
        self.exit_rules        = exit_rules or {}
        self.position_sizing   = position_sizing or {}
        self.evaluation_metric = evaluation_metric
        self.extra             = extra

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "dataset_id":        self.dataset_id,
            "symbols":           self.symbols,
            "start_date":        self.start_date,
            "end_date":          self.end_date,
            "timeframe":         self.timeframe,
            "vendor":            self.vendor,
            "initial_capital":   self.initial_capital,
            "trading_mode":      self.trading_mode,
            "indicators":        self.indicators,
            "blend_method":      self.blend_method,
            "blend_weights":     self.blend_weights,
            "phiai_enabled":     self.phiai_enabled,
            "phiai_constraints": self.phiai_constraints,
            "exit_rules":        self.exit_rules,
            "position_sizing":   self.position_sizing,
            "evaluation_metric": self.evaluation_metric,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        """Reconstruct a RunConfig from a dictionary."""
        return cls(**d)

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise ``phinance.exceptions.ConfigurationError`` on invalid fields."""
        from phinance.exceptions import ConfigurationError

        if not self.symbols:
            raise ConfigurationError("RunConfig: 'symbols' must not be empty.")
        if self.initial_capital <= 0:
            raise ConfigurationError(
                f"RunConfig: 'initial_capital' must be > 0; got {self.initial_capital}."
            )
        valid_modes = {"equities", "options"}
        if self.trading_mode not in valid_modes:
            raise ConfigurationError(
                f"RunConfig: 'trading_mode' must be one of {valid_modes}; "
                f"got '{self.trading_mode}'."
            )

    def __repr__(self) -> str:
        return (
            f"RunConfig(symbols={self.symbols}, timeframe={self.timeframe!r}, "
            f"vendor={self.vendor!r}, mode={self.trading_mode!r})"
        )

"""Adapter to load portfolio classes from `phi/backtest/portfolio.py` without package side effects."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_PORTFOLIO_PATH = Path(__file__).resolve().parents[1] / "backtest" / "portfolio.py"
_spec = spec_from_file_location("phi_backtest_portfolio_standalone", _PORTFOLIO_PATH)
_module = module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

Order = _module.Order
Portfolio = _module.Portfolio

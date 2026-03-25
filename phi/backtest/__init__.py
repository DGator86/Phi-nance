"""Phi-nance backtest engines."""

from __future__ import annotations

from phi.logging import get_logger

logger = get_logger(__name__)

from phi.backtest.direct import run_direct_backtest, run_portfolio_backtest
from phi.backtest.options_engine import OptionsBacktestEngine
from phi.backtest.payload_utils import build_regime_boosts_from_payload
from phi.backtest.vectorized_engine import VectorizedEngine


def get_engine(name: str):
    key = str(name).strip().lower()
    if key in {"vectorized", "vectorised"}:
        return VectorizedEngine()
    if key in {"options", "options_engine"}:
        return OptionsBacktestEngine()
    raise ValueError(f"Unknown backtest engine: {name}")


__all__ = [
    "run_direct_backtest",
    "run_portfolio_backtest",
    "OptionsBacktestEngine",
    "VectorizedEngine",
    "get_engine",
    "build_regime_boosts_from_payload",
]

"""Phi-nance backtest engines."""

from __future__ import annotations

from .direct import run_direct_backtest
from .engine import BacktestEngine


def get_engine(engine_type: str = "lumibot") -> BacktestEngine:
    if engine_type == "lumibot":
        from .lumibot_engine import LumibotEngine

        return LumibotEngine()
    if engine_type == "vectorized":
        from .vectorized_engine import VectorizedEngine

        return VectorizedEngine()
    raise ValueError(f"Unknown engine: {engine_type}")


__all__ = ["BacktestEngine", "get_engine", "run_direct_backtest"]

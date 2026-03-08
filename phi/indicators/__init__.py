"""
Phi-nance Indicator Compute Functions
=====================================

Lightweight signal computation from OHLCV (no Lumibot).
Used by BlendedWorkbenchStrategy for multi-indicator blending.
"""

from phi.logging import get_logger

logger = get_logger(__name__)

from .orderflow import OHLCVOrderFlowProvider, get_order_flow_provider, set_order_flow_provider
from .simple import compute_bollinger, compute_breakout, compute_dual_sma, compute_macd, compute_mean_reversion, compute_rsi
from .simple import INDICATOR_COMPUTERS

__all__ = [
    "compute_rsi",
    "compute_macd",
    "compute_bollinger",
    "compute_dual_sma",
    "compute_mean_reversion",
    "compute_breakout",
    "INDICATOR_COMPUTERS",
    "OHLCVOrderFlowProvider",
    "set_order_flow_provider",
    "get_order_flow_provider",
]

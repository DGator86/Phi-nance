"""Order flow indicator package exports and provider management."""

from __future__ import annotations

from phi.indicators.orderflow.base import OrderFlowProvider
from phi.indicators.orderflow.cumulative_delta import compute_cumulative_delta_signal
from phi.indicators.orderflow.liquidity import compute_liquidity_signal
from phi.indicators.orderflow.providers.ohlcv_provider import OHLCVOrderFlowProvider
from phi.indicators.orderflow.volume_profile import compute_volume_profile_signal
from phi.indicators.orderflow.vwap import compute_vwap_series, compute_vwap_signal

_ORDER_FLOW_PROVIDER: OrderFlowProvider = OHLCVOrderFlowProvider()


def set_order_flow_provider(provider: OrderFlowProvider) -> None:
    """Set global order flow provider used by order flow indicators."""
    global _ORDER_FLOW_PROVIDER
    _ORDER_FLOW_PROVIDER = provider


def get_order_flow_provider() -> OrderFlowProvider:
    """Get the currently configured global order flow provider."""
    return _ORDER_FLOW_PROVIDER


__all__ = [
    "OrderFlowProvider",
    "OHLCVOrderFlowProvider",
    "compute_vwap_series",
    "compute_vwap_signal",
    "compute_volume_profile_signal",
    "compute_cumulative_delta_signal",
    "compute_liquidity_signal",
    "set_order_flow_provider",
    "get_order_flow_provider",
]

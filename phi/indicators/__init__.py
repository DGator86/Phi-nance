"""Public indicator compute exports."""

from .information import (
    compute_entropy_signal,
    compute_fisher_information_signal,
    compute_kld_signal,
    compute_mutual_info_signal,
)
from .information_flow import rolling_granger_causality, rolling_transfer_entropy
from .orderflow import OHLCVOrderFlowProvider, get_order_flow_provider, set_order_flow_provider
from .simple import (
    INDICATOR_COMPUTERS,
    compute_bollinger,
    compute_breakout,
    compute_dual_sma,
    compute_fisher_information,
    compute_granger_causality,
    compute_kl_divergence,
    compute_macd,
    compute_mean_reversion,
    compute_mft_energy,
    compute_mft_signal,
    compute_mutual_information,
    compute_return_entropy,
    compute_rolling_entropy,
    compute_rsi,
    compute_transfer_entropy,
)
from phi.lob.book import OrderBook


def compute_order_flow_imbalance(book: OrderBook) -> float:
    """Compute top-of-book order flow imbalance in [-1, 1]."""
    bid = book.best_bid()
    ask = book.best_ask()
    if bid is None or ask is None:
        return 0.0
    bid_volume = float(book.bids.get(bid, 0.0))
    ask_volume = float(book.asks.get(ask, 0.0))
    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0
    return (bid_volume - ask_volume) / total


def compute_depth_ratio(book: OrderBook) -> float:
    """Return bid/ask top-level depth ratio."""
    bid = book.best_bid()
    ask = book.best_ask()
    if bid is None or ask is None:
        return 1.0
    bid_volume = float(book.bids.get(bid, 0.0))
    ask_volume = float(book.asks.get(ask, 0.0))
    return bid_volume / max(ask_volume, 1e-10)


def compute_microprice(book: OrderBook) -> float:
    """Compute microprice based on best level volumes."""
    bid = book.best_bid()
    ask = book.best_ask()
    if bid is None and ask is None:
        return 0.0
    if bid is None:
        return float(ask)
    if ask is None:
        return float(bid)
    bid_volume = float(book.bids.get(bid, 0.0))
    ask_volume = float(book.asks.get(ask, 0.0))
    total = bid_volume + ask_volume
    if total <= 0:
        return float((bid + ask) / 2.0)
    return float((ask * bid_volume + bid * ask_volume) / total)

__all__ = [
    "compute_rsi",
    "compute_macd",
    "compute_bollinger",
    "compute_dual_sma",
    "compute_mean_reversion",
    "compute_breakout",
    "compute_rolling_entropy",
    "compute_return_entropy",
    "compute_mutual_information",
    "compute_fisher_information",
    "compute_kl_divergence",
    "compute_transfer_entropy",
    "compute_granger_causality",
    "rolling_transfer_entropy",
    "rolling_granger_causality",
    "compute_mft_signal",
    "compute_mft_energy",
    "INDICATOR_COMPUTERS",
    "OHLCVOrderFlowProvider",
    "set_order_flow_provider",
    "get_order_flow_provider",
    "compute_entropy_signal",
    "compute_mutual_info_signal",
    "compute_fisher_information_signal",
    "compute_kld_signal",
    "compute_order_flow_imbalance",
    "compute_depth_ratio",
    "compute_microprice",
]

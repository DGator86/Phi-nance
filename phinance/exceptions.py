"""
Phinance custom exception hierarchy.

Raising typed exceptions lets callers handle errors precisely
without catching broad Exception.

Usage
-----
    from phinance.exceptions import DataFetchError, CacheError

    try:
        df = fetch_and_cache(...)
    except DataFetchError as exc:
        logger.error("Vendor error: %s", exc)
    except CacheError as exc:
        logger.error("Cache I/O error: %s", exc)
"""

from __future__ import annotations


# ── Base ─────────────────────────────────────────────────────────────────────


class PhinanceError(Exception):
    """Root exception for all Phi-nance errors."""


# ── Data layer ────────────────────────────────────────────────────────────────


class DataError(PhinanceError):
    """Base for data-layer errors."""


class DataFetchError(DataError):
    """Raised when a vendor fetch fails after all retries."""


class CacheError(DataError):
    """Raised when a cache read/write operation fails."""


class UnsupportedVendorError(DataError):
    """Raised when an unknown vendor key is supplied."""


class UnsupportedTimeframeError(DataError):
    """Raised when a vendor does not support the requested timeframe."""


class DataValidationError(DataError):
    """Raised when OHLCV data fails sanity checks."""


# ── Strategy layer ────────────────────────────────────────────────────────────


class StrategyError(PhinanceError):
    """Base for strategy errors."""


class UnknownIndicatorError(StrategyError):
    """Raised when an indicator name is not registered in the catalog."""


class IndicatorComputationError(StrategyError):
    """Raised when an indicator computation fails."""


# ── Blending layer ────────────────────────────────────────────────────────────


class BlendingError(PhinanceError):
    """Base for signal-blending errors."""


class UnsupportedBlendMethodError(BlendingError):
    """Raised when an unknown blend method is requested."""


# ── Optimization layer ────────────────────────────────────────────────────────


class OptimizationError(PhinanceError):
    """Base for optimization errors."""


class NoFeasibleSolutionError(OptimizationError):
    """Raised when the optimizer finds no valid parameter combination."""


# ── Backtest layer ────────────────────────────────────────────────────────────


class BacktestError(PhinanceError):
    """Base for backtest errors."""


class InsufficientDataError(BacktestError):
    """Raised when OHLCV data has too few rows to run a backtest."""


class ConfigurationError(BacktestError):
    """Raised when a RunConfig value is invalid."""


# ── Options layer ─────────────────────────────────────────────────────────────


class OptionsError(PhinanceError):
    """Base for options-mode errors."""


class OptionsDataError(OptionsError):
    """Raised when options chain data cannot be retrieved."""


# ── Storage layer ─────────────────────────────────────────────────────────────


class StorageError(PhinanceError):
    """Base for storage / persistence errors."""


class RunNotFoundError(StorageError):
    """Raised when a requested run_id does not exist on disk."""

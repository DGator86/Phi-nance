"""Custom exception hierarchy for Phi-nance."""

from __future__ import annotations


class PhiError(Exception):
    """Base exception for all Phi-nance errors."""


class DataFetchError(PhiError):
    """Raised when data cannot be fetched from a vendor."""


class CacheCorruptedError(DataFetchError):
    """Raised when cached data is corrupted."""


class ValidationError(PhiError):
    """Raised when input validation fails."""


class ConfigurationError(PhiError):
    """Raised when there is an issue with configuration."""


class BacktestError(PhiError):
    """Raised during backtest execution."""


class OptimizationError(PhiError):
    """Raised during PhiAI optimization."""

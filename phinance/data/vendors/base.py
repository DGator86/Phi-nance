"""
phinance.data.vendors.base
==========================

Abstract base class for all OHLCV data-source vendors.

New vendors should subclass ``BaseVendor`` and implement ``fetch()``.
The base class provides the ``_normalize`` and ``_sanity_check`` helpers
so every vendor delivers data in a consistent format.

Example
-------
    class MyVendor(BaseVendor):
        name = "myvendor"

        def fetch(self, symbol, timeframe, start, end, **kwargs):
            raw = _call_my_api(symbol, timeframe, start, end)
            return self._normalize(raw)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from phinance.data.cache import normalize_ohlcv, ohlcv_sanity_check
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


class BaseVendor(ABC):
    """Abstract OHLCV data-source vendor.

    Subclasses must set the ``name`` class attribute and implement
    the ``fetch`` method.

    Attributes
    ----------
    name : str
        Short identifier used in cache paths and error messages.
    """

    name: str = "base"

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch OHLCV data and return a normalised DataFrame.

        Parameters
        ----------
        symbol : str
            Ticker / pair symbol.
        timeframe : str
            One of ``"1D"``, ``"1H"``, ``"15m"``, ``"5m"``, ``"1m"``.
        start : str
            Start date ``YYYY-MM-DD``.
        end : str
            End date ``YYYY-MM-DD``.
        **kwargs
            Vendor-specific parameters (e.g. ``api_key``).

        Returns
        -------
        pd.DataFrame
            Normalised OHLCV DataFrame with columns
            ``[open, high, low, close, volume]`` and DatetimeIndex.
        """
        raise NotImplementedError

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delegate to the module-level ``normalize_ohlcv`` helper."""
        return normalize_ohlcv(df)

    def _sanity_check(self, df: pd.DataFrame, symbol: str = "") -> None:
        """Run basic sanity checks and log warnings."""
        ohlcv_sanity_check(df, symbol=symbol or self.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

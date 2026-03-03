"""
phinance.data.vendors
=====================

Pluggable vendor adapters for OHLCV data acquisition.

Each vendor module exposes a class that inherits from
``phinance.data.vendors.base.BaseVendor`` and implements the ``fetch()``
method.

Available vendors
-----------------
  YFinanceVendor      — yfinance (no API key required)
  AlphaVantageVendor  — Alpha Vantage intraday + daily
  BinanceVendor       — Binance public kline data (no API key)

Adding a new vendor
-------------------
  1. Create ``phinance/data/vendors/myvendor.py``
  2. Subclass ``BaseVendor`` and implement ``fetch()``
  3. Register the new key in ``phinance.data.cache.fetch_and_cache``
"""

from phinance.data.vendors.base import BaseVendor
from phinance.data.vendors.yfinance import YFinanceVendor
from phinance.data.vendors.alphavantage import AlphaVantageVendor
from phinance.data.vendors.binance import BinanceVendor

__all__ = [
    "BaseVendor",
    "YFinanceVendor",
    "AlphaVantageVendor",
    "BinanceVendor",
]

"""
Lazy Lumibot shim
-----------------
Importing lumibot at module-level triggers lumibot/credentials.py which
instantiates broker objects and raises an Exception if no TRADIER_TOKEN etc
is configured.

All strategy files should import Strategy and YahooDataBacktesting from
HERE instead of directly from lumibot, so that the actual lumibot import
only happens the first time a strategy class is *used* (inside _strat() /
_run_backtest()) rather than when the module is first imported by the
dashboard.
"""
from __future__ import annotations

import importlib
from typing import Any


def _lazy(module: str, attr: str) -> Any:
    mod = importlib.import_module(module)
    return getattr(mod, attr)


class _LazyStrategyMeta(type):
    """
    Metaclass that defers resolving the real lumibot.Strategy base class
    until the first time the concrete strategy class is actually
    instantiated, not when Python imports the module.
    """
    _real_strategy: type | None = None

    @classmethod
    def _get_real_strategy(mcs) -> type:
        if mcs._real_strategy is None:
            mcs._real_strategy = _lazy("lumibot.strategies.strategy", "Strategy")
        return mcs._real_strategy

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs):
        # Replace any occurrence of _LazyStrategy sentinel in bases with the
        # real Strategy class.  We can't resolve it yet at class-definition
        # time (that would defeat the whole purpose), so we store a sentinel
        # and swap it in __call__ / __init_subclass__ as needed.
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class _LazyStrategy:
    """
    Drop-in sentinel base class.  Concrete strategy classes inherit from
    this; the real lumibot.Strategy swap happens inside _run_backtest()
    via _make_real_class() below, which is called by the dashboard's
    _strat() helper.
    """
    pass


def _make_real_class(cls: type) -> type:
    """
    Given a strategy class that inherits from _LazyStrategy, return an
    equivalent class that inherits from the real lumibot Strategy instead.
    Called once per class by _strat() in dashboard.py.
    """
    RealStrategy = _lazy("lumibot.strategies.strategy", "Strategy")

    # Walk the MRO and replace _LazyStrategy with RealStrategy
    new_bases = tuple(
        RealStrategy if b is _LazyStrategy else b
        for b in cls.__bases__
    )
    if new_bases == cls.__bases__:
        # _LazyStrategy not in direct bases — nothing to swap
        return cls

    new_cls = type(cls.__name__, new_bases, dict(cls.__dict__))
    return new_cls


# Public re-exports expected by strategy files
Strategy = _LazyStrategy


def YahooDataBacktesting():
    """Lazy accessor — call as YahooDataBacktesting() to get the class."""
    return _lazy("lumibot.backtesting", "YahooDataBacktesting")


def AlphaVantageBacktesting():
    """Lazy accessor — call as AlphaVantageBacktesting() to get the class."""
    return _lazy("lumibot.backtesting", "AlphaVantageBacktesting")

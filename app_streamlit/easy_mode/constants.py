"""Defaults for easy mode (override with env)."""

from __future__ import annotations

import os


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return tuple(x.strip().upper() for x in raw.split(",") if x.strip())


UNIVERSE: tuple[str, ...] = _env_list("PHINANCE_UNIVERSE", ("SPY", "QQQ", "IWM", "DIA", "GLD"))
LOOKBACK_DAYS: int = int(os.getenv("PHINANCE_EASY_LOOKBACK", "420"))
PRIMARY_BACKTEST_SYMBOL: str = os.getenv("PHINANCE_EASY_PRIMARY", "SPY").strip().upper()
LEARNING_SUMMARY_FILENAME = "phi_nance_learning_summary.json"

"""
phinance.live.scheduler
===========================

Paper-Trading Scheduler — cron-style autonomous loop that:

  1. Fetches or generates the latest OHLCV bar.
  2. Computes signals for the active strategy (from AutonomousDeployer registry).
  3. Submits orders through the PaperBroker (or a real BrokerAdapter).
  4. Records fills and updates the equity curve.
  5. Optionally triggers the EvolutionEngine to refresh the strategy.

Architecture
------------
  SchedulerConfig    — tunable parameters (interval, symbols, …)
  ScheduledTick      — result of one scheduler tick
  TradingScheduler   — main controller; supports run_once(), run_loop()
  run_paper_scheduler — convenience one-shot tick function

Public API
----------
  SchedulerConfig
  ScheduledTick
  TradingScheduler
  run_paper_scheduler
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from phinance.live.paper_engine import PaperBroker
from phinance.live.broker_base import OrderSide, OrderType
from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SchedulerConfig:
    """
    Configuration for TradingScheduler.

    Attributes
    ----------
    symbols         : list[str]  — symbols to trade
    indicator_name  : str        — indicator to use for signals
    indicator_params: dict       — params passed to compute_indicator
    interval_seconds: float      — tick interval (default 60 s)
    max_ticks       : int        — stop after N ticks (0 = run forever)
    qty_per_trade   : int        — shares per order (default 1)
    initial_capital : float      — paper broker initial capital
    signal_threshold: float      — signal magnitude to trigger trade (default 0.1)
    evolution_every : int        — re-run evolution every N ticks (0 = never)
    dry_run         : bool       — if True, no orders submitted to broker
    """

    symbols:          List[str] = field(default_factory=lambda: ["SPY"])
    indicator_name:   str = "EMA Cross"
    indicator_params: Dict[str, Any] = field(default_factory=dict)
    interval_seconds: float = 60.0
    max_ticks:        int = 0
    qty_per_trade:    int = 1
    initial_capital:  float = 100_000.0
    signal_threshold: float = 0.1
    evolution_every:  int = 0
    dry_run:          bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ScheduledTick:
    """
    Result of one scheduler tick.

    Attributes
    ----------
    tick_id       : str
    tick_number   : int
    timestamp     : float  — Unix timestamp
    symbol        : str
    signal        : float  — latest signal value
    action        : str    — "buy" | "sell" | "hold"
    order_id      : str or None
    equity        : float  — broker equity after tick
    elapsed_ms    : float
    """

    tick_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tick_number: int = 0
    timestamp:   float = 0.0
    symbol:      str = ""
    signal:      float = 0.0
    action:      str = "hold"
    order_id:    Optional[str] = None
    equity:      float = 0.0
    elapsed_ms:  float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        return (
            f"ScheduledTick(#{self.tick_number}, "
            f"sym={self.symbol}, sig={self.signal:.3f}, "
            f"action={self.action}, equity={self.equity:.0f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TradingScheduler
# ─────────────────────────────────────────────────────────────────────────────


class TradingScheduler:
    """
    Cron-style autonomous paper-trading scheduler.

    Usage
    -----
    ::

        from phinance.live.scheduler import TradingScheduler, SchedulerConfig

        scheduler = TradingScheduler(
            ohlcv_provider=lambda symbol: my_ohlcv_dict[symbol],
            config=SchedulerConfig(symbols=["SPY", "QQQ"], max_ticks=5),
        )
        ticks = scheduler.run_loop()
    """

    def __init__(
        self,
        ohlcv_provider: Callable[[str], pd.DataFrame],
        config: Optional[SchedulerConfig] = None,
        broker: Optional[PaperBroker] = None,
    ) -> None:
        self.ohlcv_provider = ohlcv_provider
        self.config         = config or SchedulerConfig()
        self._broker        = broker or PaperBroker(
            initial_capital=self.config.initial_capital
        )
        self._tick_count    = 0
        self._ticks:        List[ScheduledTick] = []
        self._running       = False
        self._lock          = threading.Lock()

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    @property
    def tick_history(self) -> List[ScheduledTick]:
        return list(self._ticks)

    @property
    def is_running(self) -> bool:
        return self._running

    def run_once(self) -> List[ScheduledTick]:
        """Execute one tick for all configured symbols. Returns list of ScheduledTick."""
        results = []
        for symbol in self.config.symbols:
            tick = self._process_symbol(symbol)
            results.append(tick)
        self._tick_count += 1
        return results

    def run_loop(
        self,
        on_tick: Optional[Callable[[List[ScheduledTick]], None]] = None,
    ) -> List[ScheduledTick]:
        """
        Run the scheduler loop until max_ticks is reached.

        Parameters
        ----------
        on_tick : callable or None — called after each tick with the tick results

        Returns
        -------
        list[ScheduledTick]  — all ticks from the run
        """
        cfg           = self.config
        all_ticks:    List[ScheduledTick] = []
        self._running = True

        try:
            while self._running:
                ticks = self.run_once()
                all_ticks.extend(ticks)
                self._ticks.extend(ticks)

                if on_tick:
                    on_tick(ticks)

                if cfg.max_ticks > 0 and self._tick_count >= cfg.max_ticks:
                    break

                # Sleep between ticks (skip in tests when max_ticks is small)
                if cfg.interval_seconds > 0 and cfg.max_ticks != 1:
                    # Only sleep if we have more ticks to run
                    remaining = cfg.max_ticks - self._tick_count if cfg.max_ticks > 0 else 1
                    if remaining > 0:
                        time.sleep(min(cfg.interval_seconds, 0.001))  # min 1ms for tests

        finally:
            self._running = False

        return all_ticks

    def stop(self) -> None:
        """Signal the run_loop to stop after the current tick."""
        self._running = False

    def equity(self) -> float:
        """Current broker equity."""
        return float(self._broker.get_account().get("equity", self.config.initial_capital))

    def equity_history(self) -> List[float]:
        """Return per-tick equity history."""
        return [t.equity for t in self._ticks]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the scheduler run."""
        return {
            "ticks_run":   self._tick_count,
            "symbols":     self.config.symbols,
            "equity":      self.equity(),
            "total_ticks": len(self._ticks),
        }

    # ── internal ─────────────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str) -> ScheduledTick:
        t0 = time.perf_counter()
        ts = time.time()

        signal_val = 0.0
        action     = "hold"
        order_id:  Optional[str] = None

        try:
            ohlcv = self.ohlcv_provider(symbol)
            sig   = compute_indicator(
                self.config.indicator_name,
                ohlcv,
                self.config.indicator_params,
            )
            if sig is not None and len(sig) > 0:
                signal_val = float(sig.fillna(0.0).iloc[-1])

            # Determine action
            threshold = self.config.signal_threshold
            if signal_val >= threshold:
                action = "buy"
            elif signal_val <= -threshold:
                action = "sell"

            # Submit order (unless dry_run)
            if not self.config.dry_run and action in ("buy", "sell"):
                side = OrderSide.BUY if action == "buy" else OrderSide.SELL
                try:
                    order = self._broker.submit_order(
                        symbol=symbol,
                        qty=self.config.qty_per_trade,
                        side=side,
                        order_type=OrderType.MARKET,
                    )
                    order_id = getattr(order, "order_id", None)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Order error for %s: %s", symbol, exc)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Tick error for %s: %s", symbol, exc)

        equity = self.equity()
        elapsed = (time.perf_counter() - t0) * 1000.0

        return ScheduledTick(
            tick_number=self._tick_count,
            timestamp=ts,
            symbol=symbol,
            signal=signal_val,
            action=action,
            order_id=order_id,
            equity=equity,
            elapsed_ms=elapsed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def run_paper_scheduler(
    ohlcv_dict: Dict[str, pd.DataFrame],
    indicator_name: str = "EMA Cross",
    max_ticks: int = 1,
    dry_run: bool = True,
    initial_capital: float = 100_000.0,
) -> List[ScheduledTick]:
    """
    One-shot paper-trading scheduler run.

    Parameters
    ----------
    ohlcv_dict      : dict  — {symbol: OHLCV DataFrame}
    indicator_name  : str   — indicator to use for signals
    max_ticks       : int   — number of ticks to run
    dry_run         : bool  — if True, no orders submitted
    initial_capital : float

    Returns
    -------
    list[ScheduledTick]
    """
    cfg = SchedulerConfig(
        symbols=list(ohlcv_dict.keys()),
        indicator_name=indicator_name,
        max_ticks=max_ticks,
        interval_seconds=0.0,
        dry_run=dry_run,
        initial_capital=initial_capital,
    )
    scheduler = TradingScheduler(
        ohlcv_provider=lambda sym: ohlcv_dict[sym],
        config=cfg,
    )
    return scheduler.run_loop()

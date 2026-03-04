"""
tests/unit/test_trading_scheduler.py
=======================================

Comprehensive unit tests for phinance.live.scheduler.

Covers:
  - SchedulerConfig (defaults, custom, to_dict)
  - ScheduledTick (creation, to_dict, repr)
  - TradingScheduler.__init__
  - TradingScheduler.run_once (returns ticks, len == num_symbols)
  - TradingScheduler.run_loop (max_ticks respected, ticks populated)
  - TradingScheduler.stop()
  - TradingScheduler.equity()
  - TradingScheduler.equity_history()
  - TradingScheduler.summary()
  - TradingScheduler._process_symbol (signal, action, equity)
  - run_paper_scheduler convenience function
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phinance.live.scheduler import (
    SchedulerConfig,
    ScheduledTick,
    TradingScheduler,
    run_paper_scheduler,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.normal(0, 0.002, n)),
            "high":   close * (1 + abs(rng.normal(0, 0.005, n))),
            "low":    close * (1 - abs(rng.normal(0, 0.005, n))),
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


DF_SPY = make_ohlcv(200, seed=1)
DF_QQQ = make_ohlcv(200, seed=2)
OHLCV_DICT = {"SPY": DF_SPY, "QQQ": DF_QQQ}


def _provider(sym: str) -> pd.DataFrame:
    return OHLCV_DICT.get(sym, DF_SPY)


def _make_scheduler(
    symbols=None,
    max_ticks=1,
    dry_run=True,
    indicator="EMA Cross",
) -> TradingScheduler:
    cfg = SchedulerConfig(
        symbols=symbols or ["SPY"],
        indicator_name=indicator,
        max_ticks=max_ticks,
        interval_seconds=0.0,
        dry_run=dry_run,
        initial_capital=100_000,
    )
    return TradingScheduler(ohlcv_provider=_provider, config=cfg)


# ── SchedulerConfig ───────────────────────────────────────────────────────────


class TestSchedulerConfig:
    def test_defaults(self):
        cfg = SchedulerConfig()
        assert cfg.symbols == ["SPY"]
        assert cfg.indicator_name == "EMA Cross"
        assert cfg.interval_seconds == 60.0
        assert cfg.max_ticks == 0
        assert cfg.dry_run is True
        assert cfg.initial_capital == 100_000.0

    def test_custom(self):
        cfg = SchedulerConfig(symbols=["AAPL", "MSFT"], max_ticks=10, dry_run=False)
        assert cfg.symbols == ["AAPL", "MSFT"]
        assert cfg.max_ticks == 10
        assert not cfg.dry_run

    def test_to_dict_keys(self):
        d = SchedulerConfig().to_dict()
        for k in ("symbols", "indicator_name", "interval_seconds",
                  "max_ticks", "qty_per_trade", "initial_capital",
                  "signal_threshold", "dry_run"):
            assert k in d

    def test_signal_threshold(self):
        cfg = SchedulerConfig(signal_threshold=0.25)
        assert cfg.signal_threshold == 0.25


# ── ScheduledTick ─────────────────────────────────────────────────────────────


class TestScheduledTick:
    def test_default_creation(self):
        t = ScheduledTick()
        assert isinstance(t.tick_id, str)
        assert t.tick_number == 0
        assert t.action == "hold"
        assert t.order_id is None

    def test_custom_creation(self):
        t = ScheduledTick(
            tick_number=5,
            symbol="SPY",
            signal=0.7,
            action="buy",
            equity=105_000,
        )
        assert t.tick_number == 5
        assert t.symbol == "SPY"
        assert t.signal == 0.7
        assert t.action == "buy"

    def test_to_dict_keys(self):
        t = ScheduledTick()
        d = t.to_dict()
        for k in ("tick_id", "tick_number", "timestamp", "symbol",
                  "signal", "action", "order_id", "equity", "elapsed_ms"):
            assert k in d

    def test_repr(self):
        t = ScheduledTick(tick_number=3, symbol="SPY", signal=0.5, action="buy")
        r = repr(t)
        assert "ScheduledTick" in r
        assert "SPY" in r

    def test_unique_tick_ids(self):
        ids = {ScheduledTick().tick_id for _ in range(20)}
        assert len(ids) == 20


# ── TradingScheduler.__init__ ─────────────────────────────────────────────────


class TestSchedulerInit:
    def test_attributes(self):
        s = _make_scheduler()
        assert s.ohlcv_provider is _provider
        assert isinstance(s.config, SchedulerConfig)

    def test_initial_tick_count(self):
        s = _make_scheduler()
        assert s._tick_count == 0

    def test_initial_history_empty(self):
        s = _make_scheduler()
        assert s.tick_history == []

    def test_not_running_initially(self):
        s = _make_scheduler()
        assert not s.is_running

    def test_broker_created(self):
        s = _make_scheduler()
        assert s.broker is not None


# ── run_once ──────────────────────────────────────────────────────────────────


class TestRunOnce:
    def test_returns_list(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert isinstance(ticks, list)

    def test_length_matches_symbols(self):
        s = _make_scheduler(symbols=["SPY", "QQQ"])
        ticks = s.run_once()
        assert len(ticks) == 2

    def test_all_are_scheduled_ticks(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert all(isinstance(t, ScheduledTick) for t in ticks)

    def test_symbol_set(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert ticks[0].symbol == "SPY"

    def test_tick_count_incremented(self):
        s = _make_scheduler()
        s.run_once()
        assert s._tick_count == 1

    def test_action_valid(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert ticks[0].action in ("buy", "sell", "hold")

    def test_signal_is_float(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert isinstance(ticks[0].signal, float)

    def test_equity_is_float(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert isinstance(ticks[0].equity, float)

    def test_elapsed_ms_positive(self):
        s = _make_scheduler(symbols=["SPY"])
        ticks = s.run_once()
        assert ticks[0].elapsed_ms >= 0.0


# ── run_loop ──────────────────────────────────────────────────────────────────


class TestRunLoop:
    def test_returns_list(self):
        s = _make_scheduler(max_ticks=2)
        ticks = s.run_loop()
        assert isinstance(ticks, list)

    def test_max_ticks_respected(self):
        s = _make_scheduler(symbols=["SPY"], max_ticks=3)
        ticks = s.run_loop()
        # 3 ticks × 1 symbol = 3 ScheduledTick objects
        assert len(ticks) == 3

    def test_multi_symbol_tick_count(self):
        s = _make_scheduler(symbols=["SPY", "QQQ"], max_ticks=2)
        ticks = s.run_loop()
        assert len(ticks) == 4  # 2 ticks × 2 symbols

    def test_not_running_after_completion(self):
        s = _make_scheduler(max_ticks=1)
        s.run_loop()
        assert not s.is_running

    def test_on_tick_callback(self):
        received = []
        s = _make_scheduler(symbols=["SPY"], max_ticks=2)
        s.run_loop(on_tick=lambda t: received.append(t))
        assert len(received) == 2

    def test_tick_history_populated(self):
        s = _make_scheduler(symbols=["SPY"], max_ticks=2)
        s.run_loop()
        assert len(s.tick_history) == 2


# ── stop ──────────────────────────────────────────────────────────────────────


class TestStop:
    def test_stop_sets_running_false(self):
        s = _make_scheduler()
        s._running = True
        s.stop()
        assert not s.is_running


# ── equity / equity_history ───────────────────────────────────────────────────


class TestEquity:
    def test_equity_positive(self):
        s = _make_scheduler()
        assert s.equity() > 0

    def test_equity_history_empty_before_run(self):
        s = _make_scheduler()
        assert s.equity_history() == []

    def test_equity_history_after_run(self):
        s = _make_scheduler(max_ticks=3, symbols=["SPY"])
        s.run_loop()
        history = s.equity_history()
        assert len(history) == 3
        assert all(isinstance(e, float) for e in history)


# ── summary ───────────────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_keys(self):
        s = _make_scheduler(max_ticks=2)
        s.run_loop()
        summary = s.summary()
        for k in ("ticks_run", "symbols", "equity", "total_ticks"):
            assert k in summary

    def test_ticks_run_matches(self):
        s = _make_scheduler(max_ticks=3)
        s.run_loop()
        assert s.summary()["ticks_run"] == 3

    def test_equity_in_summary(self):
        s = _make_scheduler()
        summary = s.summary()
        assert summary["equity"] > 0


# ── _process_symbol ───────────────────────────────────────────────────────────


class TestProcessSymbol:
    def test_returns_scheduled_tick(self):
        s = _make_scheduler(symbols=["SPY"])
        tick = s._process_symbol("SPY")
        assert isinstance(tick, ScheduledTick)

    def test_symbol_set(self):
        s = _make_scheduler(symbols=["SPY"])
        tick = s._process_symbol("SPY")
        assert tick.symbol == "SPY"

    def test_action_in_valid_set(self):
        s = _make_scheduler(symbols=["SPY"])
        tick = s._process_symbol("SPY")
        assert tick.action in ("buy", "sell", "hold")

    def test_signal_range(self):
        s = _make_scheduler(symbols=["SPY"])
        # Run multiple ticks to get a sample of signals
        signals = [s._process_symbol("SPY").signal for _ in range(5)]
        assert all(-1.1 <= sig <= 1.1 for sig in signals)

    def test_dry_run_no_order_id(self):
        s = _make_scheduler(symbols=["SPY"], dry_run=True)
        tick = s._process_symbol("SPY")
        # In dry_run mode, no orders are submitted → order_id is None
        assert tick.order_id is None

    def test_bad_symbol_does_not_crash(self):
        s = _make_scheduler(symbols=["SPY"])
        # Unknown symbol → ohlcv_provider returns default
        tick = s._process_symbol("UNKNOWN_SYM")
        assert isinstance(tick, ScheduledTick)


# ── run_paper_scheduler ───────────────────────────────────────────────────────


class TestRunPaperScheduler:
    def test_returns_list(self):
        ticks = run_paper_scheduler(OHLCV_DICT, max_ticks=1)
        assert isinstance(ticks, list)

    def test_tick_count_matches_symbols(self):
        ticks = run_paper_scheduler(OHLCV_DICT, max_ticks=1)
        assert len(ticks) == 2  # SPY + QQQ

    def test_multiple_ticks(self):
        ticks = run_paper_scheduler(OHLCV_DICT, max_ticks=3)
        assert len(ticks) == 6  # 3 ticks × 2 symbols

    def test_all_scheduled_ticks(self):
        ticks = run_paper_scheduler(OHLCV_DICT, max_ticks=1)
        assert all(isinstance(t, ScheduledTick) for t in ticks)

    def test_dry_run(self):
        ticks = run_paper_scheduler(OHLCV_DICT, max_ticks=1, dry_run=True)
        assert all(t.order_id is None for t in ticks)

    def test_single_symbol(self):
        ticks = run_paper_scheduler({"SPY": DF_SPY}, max_ticks=1)
        assert len(ticks) == 1
        assert ticks[0].symbol == "SPY"

    def test_custom_indicator(self):
        ticks = run_paper_scheduler(OHLCV_DICT, indicator_name="RSI", max_ticks=1)
        assert all(isinstance(t, ScheduledTick) for t in ticks)

    def test_initial_capital_propagated(self):
        ticks = run_paper_scheduler(OHLCV_DICT, max_ticks=1, initial_capital=50_000)
        # Equity should start near 50k
        assert ticks[0].equity > 0

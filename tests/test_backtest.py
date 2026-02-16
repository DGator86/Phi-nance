"""Run a minimal backtest with synthetic bars (no Polygon/Tradier data)."""

import pandas as pd
import pytest

from phinence.assignment.engine import AssignmentEngine
from phinence.composer.composer import Composer
from phinence.engines.hedge import HedgeEngine
from phinence.engines.liquidity import LiquidityEngine
from phinence.engines.regime import RegimeEngine
from phinence.engines.sentiment import SentimentEngine
from phinence.store.memory_store import InMemoryBarStore
from phinence.validation.backtest_runner import make_synthetic_bars, run_backtest_fold
from phinence.validation.walk_forward import WFMode, WFWindow, WalkForwardHarness, expanding_windows


def test_make_synthetic_bars() -> None:
    df = make_synthetic_bars("SPY", "2024-01-01", "2024-01-31", bars_per_day=30, seed=42)
    assert len(df) >= 30 * 20  # ~20 business days
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_backtest_fold_returns_metrics() -> None:
    df = make_synthetic_bars("SPY", "2023-06-01", "2024-06-30", bars_per_day=100, seed=123)
    store = InMemoryBarStore()
    store.put_1m_bars("SPY", df)
    assigner = AssignmentEngine(store)
    composer = Composer()
    engines = {
        "liquidity": LiquidityEngine(),
        "regime": RegimeEngine(),
        "sentiment": SentimentEngine(),
        "hedge": HedgeEngine(),
    }
    start = pd.Timestamp("2023-06-01")
    end = pd.Timestamp("2024-06-30")
    folds = list(expanding_windows(start, end, WFMode.DAILY))
    assert len(folds) >= 1
    fold = folds[0]
    from phinence.contracts.projection_packet import Horizon
    result = run_backtest_fold(fold, "SPY", store, assigner, engines, composer, horizon=Horizon.DAILY)
    assert "oos_auc" in result
    assert "cone_50" in result
    assert "cone_75" in result
    assert "cone_90" in result
    assert 0 <= result["oos_auc"] <= 1
    assert result["n_obs"] >= 0


def test_walk_forward_harness_run_fold() -> None:
    df = make_synthetic_bars("QQQ", "2023-09-01", "2024-03-01", bars_per_day=80, seed=1)
    store = InMemoryBarStore()
    store.put_1m_bars("QQQ", df)
    harness = WalkForwardHarness(mode=WFMode.DAILY)
    assigner = AssignmentEngine(store)
    composer = Composer()
    engines = {
        "liquidity": LiquidityEngine(),
        "regime": RegimeEngine(),
        "sentiment": SentimentEngine(),
        "hedge": HedgeEngine(),
    }
    start = pd.Timestamp("2023-09-01")
    end = pd.Timestamp("2024-03-01")
    folds = list(expanding_windows(start, end, WFMode.DAILY))
    if not folds:
        pytest.skip("Not enough range for a fold")
    metrics = harness.run_fold(folds[0], "QQQ", store, assigner, engines, composer)
    assert "oos_auc" in metrics
    assert metrics.get("n_obs", 0) >= 0

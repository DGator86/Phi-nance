from scripts.profile_backtest import make_synthetic_ohlcv
from phinance.backtest.runner import run_backtest
from phinance.utils.performance import PerformanceTracker, track_time


def test_backtest_profiling_smoke() -> None:
    tracker = PerformanceTracker()
    ohlcv = make_synthetic_ohlcv(300)

    with track_time(tracker, "run_backtest"):
        result = run_backtest(
            ohlcv=ohlcv,
            symbol="SPY",
            indicators={"RSI": {"enabled": True, "params": {"period": 14}}},
            blend_weights={"RSI": 1.0},
            blend_method="weighted_sum",
            initial_capital=100_000,
        )

    assert result.total_trades >= 0
    summary = tracker.summary()
    assert "run_backtest" in summary
    assert summary["run_backtest"]["count"] == 1.0

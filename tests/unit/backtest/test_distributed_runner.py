import numpy as np
import pandas as pd
import pytest

from phinance.backtest.distributed_runner import DistributedBacktestRunner


def _synthetic_inputs(n: int = 80):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, size=n))
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )
    signal = pd.Series(np.sin(np.linspace(0, 8, n)), index=idx)
    return ohlcv, signal


def test_distributed_runner_sequential_fallback_runs_configs():
    ohlcv, signal = _synthetic_inputs()
    runner = DistributedBacktestRunner(enabled=False)

    outputs = runner.run_parallel(
        [
            {"engine": "vectorized", "ohlcv": ohlcv, "signal": signal, "symbol": "A"},
            {"engine": "vectorized", "ohlcv": ohlcv, "signal": signal * -1, "symbol": "B"},
        ]
    )

    assert len(outputs) == 2
    assert all(item["status"] == "ok" for item in outputs)
    assert outputs[0]["result"]["symbol"] == "A"


def test_distributed_runner_ray_local_mode_if_available():
    pytest.importorskip("ray")

    ohlcv, signal = _synthetic_inputs()
    runner = DistributedBacktestRunner(enabled=True, use_ray=True, local_mode=True, num_cpus=1)
    outputs = runner.run_parallel(
        [{"engine": "vectorized", "ohlcv": ohlcv, "signal": signal, "symbol": "RAY"}]
    )
    runner.shutdown()

    assert len(outputs) == 1
    assert outputs[0]["status"] == "ok"
    assert outputs[0]["result"]["symbol"] == "RAY"

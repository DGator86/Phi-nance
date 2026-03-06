import numpy as np
import pandas as pd
import pytest

from phinance.backtest.distributed_runner import DistributedBacktestRunner
from phinance.backtest.vectorized import run_vectorized_backtest


def _ohlcv(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    close = 120 + np.cumsum(np.random.default_rng(9).normal(0, 0.8, size=n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(2000, 3000, n),
        },
        index=idx,
    )


def test_parallel_results_match_sequential():
    pytest.importorskip("ray")

    ohlcv = _ohlcv()
    configs = []
    sequential = []
    for i in range(10):
        signal = pd.Series(np.sin(np.linspace(0, 8 + i * 0.2, len(ohlcv))), index=ohlcv.index)
        configs.append({"engine": "vectorized", "ohlcv": ohlcv, "signal": signal, "symbol": f"S{i}"})
        sequential.append(run_vectorized_backtest(ohlcv, signal=signal, symbol=f"S{i}").to_dict())

    runner = DistributedBacktestRunner(enabled=True, use_ray=True, local_mode=True, num_cpus=2)
    distributed = runner.run_parallel(configs)
    runner.shutdown()

    assert len(distributed) == len(sequential)
    for dist, seq in zip(distributed, sequential):
        assert dist["status"] == "ok"
        assert dist["result"]["symbol"] == seq["symbol"]
        assert dist["result"]["total_return"] == pytest.approx(seq["total_return"], rel=1e-8)
        assert dist["result"]["sharpe_ratio"] == pytest.approx(seq["sharpe_ratio"], rel=1e-8)

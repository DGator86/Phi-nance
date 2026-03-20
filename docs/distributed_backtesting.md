# Distributed Backtesting with Ray

Phi-nance supports parallel execution of independent backtests using Ray via `DistributedBacktestRunner`.

## Install

Ray is included in dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

`configs/distributed_config.yaml`:

```yaml
distributed:
  enabled: false
  use_ray: true
  num_cpus: 4
  address: auto
  timeout_s: 120
```

- Keep `enabled: false` for backward-compatible sequential mode.
- Set `address` to `ray://host:10001` for a remote cluster.

## Usage

```python
from phinance.backtest.distributed_runner import DistributedBacktestRunner

runner = DistributedBacktestRunner(enabled=True, num_cpus=4)
results = runner.run_parallel(configs)
runner.shutdown()
```

Each config should include `engine` and inputs:

- `engine: vectorized` with `ohlcv` + `signal`
- `engine: event` with `ohlcv` + indicator/blend config

## GP Integration

`phinance.meta.genetic.GPConfig` now has distributed settings. When enabled, populations are evaluated in batch through Ray-backed backtests.

## Tuning notes

- Start with `num_cpus` equal to physical cores.
- Use larger populations/sweeps to benefit from parallelism.
- Ensure configs/results are serialisable (dict, list, pandas, numpy).

## Limitations

- Ray is optional; missing Ray falls back to sequential execution.
- Very small sweeps can be slower because of scheduling overhead.

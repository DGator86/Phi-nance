# RL Training Optimisation (Step 4)

## Scope

This document summarises profiling and opt-in optimisation work for RL training loops (Execution, Strategy R&D, Risk Monitor, and hierarchical Meta agent).

## Baseline profiling

Profiling was run against the execution fallback training loop for a short smoke run (`episodes_smoke=2`) using `phinance.utils.performance.PerformanceTracker`.

Baseline timing snapshot:

| name | count | total (s) | avg (s) | min (s) | max (s) |
|---|---:|---:|---:|---:|---:|
| env_step | 17 | 0.005581 | 0.000328 | 0.000148 | 0.000410 |
| episode | 2 | 0.016749 | 0.008374 | 0.006655 | 0.010094 |
| policy_forward | 17 | 0.009449 | 0.000556 | 0.000219 | 0.005379 |

Primary bottlenecks in this setup were policy forward pass and environment stepping.

## Optimisations implemented

### 1) Parallel rollouts with Ray

- Added `RayEnvRunner` with configurable worker count and local mode for tests.
- Collects transitions in parallel and merges them into a batched structure.
- Fully opt-in via config `rl_optimisation.parallel_rollouts.enabled`.
- Safe fallback: if disabled, existing single-env flow is used.

### 2) Optimised experience buffer

- Added `OptimisedExperienceBuffer` with preallocated NumPy circular storage.
- Supports batched inserts and optional async prefetch sampling.
- Used in execution fallback loop when enabled.

### 3) GPU + mixed precision utilities

- Added reusable helpers in `phinance.rl.training_utils`:
  - device resolution (`cpu`/`cuda`)
  - policy transfer to device
  - autocast context and GradScaler fallback
- Training scripts now accept optimisation config and can move policies to GPU when enabled.

### 4) Caching repeated computations

- Added `state_cache` decorator helper.
- Execution environment optionally enables observation caching (`enable_state_cache`).

### 5) Numba acceleration path

- Added a numerical step kernel helper with Numba-JIT path and pure-Python fallback.
- Execution environment uses this kernel only when `enable_numba=true`.

### 6) Config-driven, backward-compatible integration

- Added `configs/rl_optimisation_config.yaml` with all optimisation flags disabled by default.
- Existing behavior remains unchanged unless flags are enabled.

## Post-optimisation profile snapshot

With experience buffer + prefetch + numba + cache enabled:

| name | count | total (s) | avg (s) | min (s) | max (s) |
|---|---:|---:|---:|---:|---:|
| env_step | 10 | 0.003063 | 0.000306 | 0.000150 | 0.000415 |
| episode | 2 | 0.007327 | 0.003663 | 0.001838 | 0.005488 |
| policy_forward | 10 | 0.002598 | 0.000260 | 0.000223 | 0.000302 |

Observed in this smoke benchmark: roughly ~2.3x lower total episode time.

## How to enable

1. Edit `configs/rl_optimisation_config.yaml` and flip relevant `enabled` fields to `true`.
2. Run training scripts with `--optim-config configs/rl_optimisation_config.yaml`.

Examples:

```bash
python scripts/train_execution_agent.py --fallback --optim-config configs/rl_optimisation_config.yaml
python scripts/train_strategy_rd_agent.py --fallback --optim-config configs/rl_optimisation_config.yaml
python scripts/train_risk_monitor_agent.py --fallback --optim-config configs/rl_optimisation_config.yaml
python scripts/train_meta_agent.py --fallback --optim-config configs/rl_optimisation_config.yaml
```

## Troubleshooting

- **Ray unavailable**: keep `parallel_rollouts.enabled=false` or install `ray[default]`.
- **No GPU available**: set `gpu.enabled=false` or mixed precision will be auto-bypassed.
- **Numba unavailable**: environment falls back to Python kernel.
- **AReaL absent**: use `--fallback` mode in scripts for smoke training.

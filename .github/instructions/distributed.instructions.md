---
applyTo: "phinance/backtest/distributed_runner.py,phinance/meta/distributed_fitness.py,configs/distributed_config.yaml,docs/distributed_backtesting.md"
---

# Distributed Backtesting Instructions

- Prefer serialisable payloads only in distributed tasks.
- Keep a sequential fallback path for all distributed features.
- Expose simple toggles through config (`enabled`, `use_ray`, `num_cpus`, `address`).
- Use `ray.init(local_mode=True)` for testability in CI-like environments.

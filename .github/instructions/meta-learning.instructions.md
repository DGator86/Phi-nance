# Meta-learning Instructions

- Prefer GP search (`phinance.meta.search`) for interpretability.
- Keep fitness evaluation vectorized and cache duplicate trees.
- Persist top-K individuals to the strategy vault.
- Ensure discovered strategies can be consumed by `StrategyRDAgent` and `MetaOrchestrator`.

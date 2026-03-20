# Hierarchical RL extension instructions

When adding a new option to the hierarchy:

1. Implement initiation + termination conditions in `phinance/rl/hierarchical/options.py`.
2. Wrap the low-level agent in `phinance/rl/hierarchical/wrappers.py` with a uniform `act(state, deterministic)` method.
3. Register the option in `build_default_options`.
4. Add tests in `tests/unit/rl/hierarchical/` to validate lifecycle rules.
5. If option metadata changes meta-state semantics, update `phinance/rl/hierarchical/meta_env.py`.

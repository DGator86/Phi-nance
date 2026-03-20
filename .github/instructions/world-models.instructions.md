# World Models Development Instructions

- Keep world model components under `phinance/world/`.
- Prefer small, composable modules: model, trainer, env, planner, integration.
- Maintain fallback behavior to model-free RL when model confidence is low.
- Preserve compatibility with optional AReaL installation.
- Add/maintain unit tests for model forward pass, trainer, and imagination env.

# Feature Engineering Instructions

- Favor incremental updates over full retraining where possible.
- Always write discovered features to `feature_registry.json`.
- Ensure expression evaluation is numerically safe (protected divide/log/sqrt).
- Keep feature generation deterministic with explicit random seeds in experiments.
- For live paths, reuse cached windows and avoid external API calls inside feature extractors.

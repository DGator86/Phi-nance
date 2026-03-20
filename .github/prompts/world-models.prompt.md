Implement and iterate on the Phi-nance world-model stack.

Checklist:
1. Train RSSM world model on market transitions.
2. Validate with reconstruction/reward/KL metrics.
3. Use `ImaginationEnv` for sample-efficient policy learning.
4. Use CEM planning where needed for short-horizon action optimization.
5. Fallback to model-free RL when confidence < threshold.

Prefer computationally-efficient defaults suitable for commodity hardware.

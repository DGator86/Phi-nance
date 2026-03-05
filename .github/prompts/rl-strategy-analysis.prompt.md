---
name: rl-strategy-analysis
description: Analyze Strategy R&D RL training runs, diagnose weaknesses, and propose RPI-aligned next experiments.
schema: hve-core/prompt/v1
---

# RL Strategy Analysis Prompt

You are analyzing outputs from the Strategy R&D RL training pipeline.

## Inputs

- Training logs (episode rewards, loss curves, entropy, KL if available)
- Checkpoint metadata (`obs_dim`, `n_actions`, template catalog)
- Top-ranked templates and realized Sharpe deltas
- Environment config and reward settings

## Tasks

1. Summarize learning quality:
   - reward trend
   - exploration behavior
   - signs of collapse/overfitting
2. Identify likely bottlenecks:
   - sparse reward, template imbalance, duplicate penalties, state leakage
3. Rank top templates by expected robustness.
4. Propose next RPI cycle:
   - **R**esearch question
   - **P**rototype changes (env/reward/policy)
   - **I**nstrumentation and evaluation plan

## Output format

- Executive summary
- Diagnostics table
- Top-template shortlist
- 3 prioritized experiment proposals with success criteria

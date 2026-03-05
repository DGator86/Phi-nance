# RL Strategy R&D Agent

This module adds a reinforcement-learning workflow that explores a discrete library of strategy templates and learns to prioritize templates that outperform a benchmark.

## Components

- `phinance/rl/strategy_rd_env.py`
  - `StrategyRDEnv` is a discrete action environment.
  - Action = index into precomputed strategy templates generated from indicator grids.
  - State = regime one-hot + volatility + recent reward + exploration count + best Sharpe so far.
  - Reward = `(template_sharpe - benchmark_sharpe)` with a small duplicate-action penalty.
- `scripts/train_strategy_rd_agent.py`
  - Trains with AReaL `AsyncTrainer` if installed.
  - Supports `--fallback` smoke training that samples actions randomly and writes a checkpoint.
- `configs/strategy_rd_agent.yaml`
  - Defines indicator template catalog and training hyperparameters.
- `phinance/agents/strategy_rd.py`
  - `StrategyRDAgent` loads the trained policy and returns strategy templates.
  - If policy is absent, falls back to random template selection.
- `tests/unit/rl/test_strategy_rd_env.py`
  - Unit coverage for env reset/step, reward lookup consistency, normalization, and agent fallback.

## Template Library

The first version uses a pure discrete setup for reliability and speed:

- RSI (`period`, `oversold`, `overbought`)
- Dual SMA (`fast`, `slow`)
- Bollinger (`window`, `num_std`)

Templates are generated from cartesian products of each indicator parameter grid.

## Training

Fallback smoke run (recommended first):

```bash
source venv/bin/activate
python scripts/train_strategy_rd_agent.py --config configs/strategy_rd_agent.yaml --fallback
```

AReaL run:

```bash
python scripts/train_strategy_rd_agent.py --config configs/strategy_rd_agent.yaml
```

Default checkpoint path:

- `models/strategy_rd_agent/latest.pt`

## Runtime Usage

```python
from phinance.agents.strategy_rd import StrategyRDAgent

agent = StrategyRDAgent(use_rl=True, policy_path="models/strategy_rd_agent/latest.pt")
proposal = agent.propose_strategy(
    {
        "regime": "bull",
        "volatility": 0.3,
        "recent_performance": 0.1,
        "exploration_count": 0.2,
        "best_sharpe": 0.5,
    }
)
print(proposal)
```

## Extending the Environment

- Add indicators by extending `indicator_catalog` in YAML.
- Replace synthetic/quick backtest logic with full backtest integration.
- Add regime-engine features and broader cross-sectional state features.
- Introduce hierarchical policies for indicator selection + continuous tuning.

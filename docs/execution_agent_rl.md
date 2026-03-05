# RL Execution Agent

This module introduces an RL-driven execution policy to replace static TWAP/VWAP slicing when desired.

## Components

- `phinance/rl/execution_env.py`
  - `ExecutionEnv` provides a continuous-control environment with normalized state features.
  - Action space is `[fraction_to_trade, urgency]` in `[0,1]`.
  - Reward is negative implementation shortfall against arrival price.
  - Includes linear impact model: `impact = k * (volume_traded / ADV) * spread`.
- `scripts/train_execution_agent.py`
  - Uses AReaL `AsyncTrainer` with PPO when installed.
  - Includes a fallback smoke trainer (`--fallback`) for local validation when AReaL is not present.
- `configs/execution_agent.yaml`
  - Hyperparameters and environment defaults.
- `phinance/agents/execution.py`
  - `ExecutionAgent` loads a trained policy checkpoint and outputs per-step execution decisions.
  - Fallback to TWAP is automatic if RL policy is missing or disabled.

## Training

```bash
source venv/bin/activate
python scripts/train_execution_agent.py --config configs/execution_agent.yaml --fallback
```

When AReaL is installed, omit `--fallback` to use asynchronous PPO training:

```bash
python scripts/train_execution_agent.py --config configs/execution_agent.yaml
```

Checkpoints are saved to `models/execution_agent/latest.pt` by default.

## Runtime usage

```python
from phinance.agents.execution import ExecutionAgent

agent = ExecutionAgent(use_rl=True, policy_path="models/execution_agent/latest.pt")
decision = agent.execute_order(order_payload, market_df)
print(decision.shares_to_trade, decision.urgency)
```

If the checkpoint is absent, the agent uses TWAP-style slicing automatically.

## Testing

Unit tests cover:
- environment reset/step flow,
- normalized observation range,
- execution fallback behavior.

Run:

```bash
pytest tests/unit/rl/test_execution_env.py
```

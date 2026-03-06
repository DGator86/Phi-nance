# Risk Monitor Agent (RL)

The Risk Monitor Agent is a discrete-action reinforcement-learning component that maps portfolio and market conditions into one of five predefined risk profiles.

## Environment design

- **State**: drawdown, VaR(95), beta, Greeks placeholders, market regime one-hot, volatility, correlation, leverage, rebalance age, and exploration count.
- **Actions**: 5 risk profiles from ultra-conservative to aggressive-with-hedge.
- **Reward**: incremental change in `sharpe - penalty * max_drawdown^2`.
- **Termination**: fixed horizon (default 252 steps) or catastrophic drawdown breach.

## Training

```bash
python scripts/train_risk_monitor_agent.py --config configs/risk_monitor_agent.yaml --fallback
```

For AReaL training, remove `--fallback` after installing `areal`.

## Runtime usage

```python
from phinance.agents.risk_monitor import RiskMonitorAgent

agent = RiskMonitorAgent(use_rl=True)
limits = agent.get_risk_limits(portfolio_state, market_data)
```

If no checkpoint exists, the agent automatically falls back to the moderate profile.

## Extending

- Add additional profiles in `RISK_PROFILES`.
- Replace synthetic Greeks placeholders with live options exposures once options risk feeds are available.
- Expand to multi-asset portfolio simulation through `PortfolioSimulator`.


## Advanced architectures

Training now supports configurable policy backbones via YAML:

- `model.architecture`: `mlp`, `lstm`, or `transformer`
- `model.sequence_length`: sequence window used for recurrent/transformer models

These settings are persisted in checkpoints and reused by the runtime agents.

# Hierarchical RL in Phi-nance

This phase introduces a meta-agent that selects among low-level options:

- `execution`: handles active order slicing.
- `strategy_rd`: proposes strategy templates on schedule.
- `risk_monitor`: maintains risk profile continuously.
- `idle`: explicit no-op option.

## Architecture

- `phinance/rl/hierarchical/options.py`: option contract + initiation/termination rules.
- `phinance/rl/hierarchical/meta_env.py`: wraps a base env with option-level actions.
- `phinance/rl/hierarchical/meta_agent.py`: categorical policy over options.
- `phinance/rl/hierarchical/training.py`: AReaL trainer + fallback smoke loop.
- `phinance/agents/meta_orchestrator.py`: runtime option selection in live trading.

## Training

```bash
python scripts/train_meta_agent.py --config configs/meta_agent.yaml --fallback
```

Use `--fallback` for smoke training without AReaL. Remove the flag to train with AReaL `AsyncTrainer`.

## Live usage

Set in `configs/live_config.yaml`:

```yaml
use_hierarchical: true
meta_agent_checkpoint: models/meta_agent/latest.pt
```

Then initialize `LiveEngine` with this config and attach `MetaOrchestrator` via `set_meta_orchestrator`.

## Extending options

Add new option definitions in `options.py` and option wrappers in `wrappers.py`, then include them in `build_default_options`.

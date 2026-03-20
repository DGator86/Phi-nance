# World Models & Model-Based RL (Phase 5)

This phase introduces a recurrent state-space world model (RSSM-inspired) to Phi-nance for model-based reinforcement learning.

## Why world models

A world model learns market transition dynamics in latent space:

- Predict next observation from current latent state and action.
- Predict step reward and termination probability.
- Support imagined rollouts for policy training and planning.

Benefits:

- Fewer expensive interactions with real environments.
- Counterfactual simulation and what-if planning.
- Better sample efficiency when data is limited.

## Architecture

`phinance/world/model.py` implements `WorldModelRSSM`:

- Deterministic pathway: `GRUCell` over previous stochastic state + action.
- Stochastic pathway: diagonal Gaussian prior/posterior heads.
- Decoder heads:
  - Observation reconstruction
  - Reward prediction
  - Done logit prediction

Training objective combines:

- Observation reconstruction MSE
- Reward MSE
- Done BCE-with-logits
- KL(q posterior || p prior)

## Components

- `phinance/world/model.py`: RSSM module + latent state container.
- `phinance/world/trainer.py`: training utilities and checkpointing.
- `phinance/world/imagination_env.py`: gym-like imagined environment.
- `phinance/world/planner.py`: CEM planner for action-sequence optimization.
- `phinance/world/integration.py`: adapters and confidence-based fallback.

## ExecutionAgent integration

`scripts/train_execution_agent.py` accepts:

- `--world-model-path`: optional checkpoint path.

Behavior:

1. If checkpoint exists, build `ImaginationEnv` from world model.
2. Estimate confidence from recent losses.
3. If confidence is below threshold, fallback to real model-free `ExecutionEnv`.

This keeps training robust when the world model is uncertain.

## Configuration

`configs/world_model_config.yaml` includes:

- Model dimensions (`obs_dim`, `action_dim`, `det_dim`, `stoch_dim`)
- Trainer hyperparameters (`learning_rate`, `kl_coef`, etc.)
- World-model runtime controls (`imagination_horizon`, `confidence_threshold`)

## Notes

- Current implementation is intentionally lightweight and test-friendly.
- It can be extended with multi-step sequence training, value imagination, and tighter AReaL hooks.

import numpy as np

from phinance.world.imagination_env import ImaginationEnv
from phinance.world.model import RSSMConfig, WorldModelRSSM
from phinance.world.planner import CEMPlanner
from phinance.world.trainer import WorldModelTrainer, build_transition_batch


def test_world_model_training_and_planning_loop() -> None:
    rng = np.random.default_rng(11)
    n = 128
    obs_dim = 6
    action_dim = 2

    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    actions = rng.uniform(0.0, 1.0, size=(n, action_dim)).astype(np.float32)
    next_obs = (0.9 * obs + 0.1 * actions.mean(axis=1, keepdims=True)).astype(np.float32)
    rewards = (next_obs[:, 0] - np.abs(actions[:, 0] - 0.5)).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)

    model = WorldModelRSSM(RSSMConfig(obs_dim=obs_dim, action_dim=action_dim, det_dim=32, stoch_dim=8, hidden_dim=64))
    trainer = WorldModelTrainer(model=model)

    batch = build_transition_batch(obs, actions, next_obs, rewards, dones)
    pre_loss = trainer.compute_loss(batch)["loss"].item()
    for _ in range(20):
        trainer.train_step(batch)
    post_loss = trainer.compute_loss(batch)["loss"].item()

    assert post_loss < pre_loss

    env = ImaginationEnv(model=model, initial_observation=obs[0])
    state = env.reset()
    assert state.shape == (obs_dim,)

    planner = CEMPlanner(model=model)
    latent = model.initial_state(batch_size=1)
    action = planner.plan(latent)
    assert action.shape == (action_dim,)
    assert np.isfinite(action).all()

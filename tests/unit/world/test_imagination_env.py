import numpy as np

from phinance.world.imagination_env import ImaginationEnv, ImaginationEnvConfig
from phinance.world.model import RSSMConfig, WorldModelRSSM


def test_imagination_env_rollout_finite() -> None:
    model = WorldModelRSSM(RSSMConfig(obs_dim=9, action_dim=2, det_dim=32, stoch_dim=8, hidden_dim=32))
    env = ImaginationEnv(
        model=model,
        initial_observation=np.zeros(9, dtype=np.float32),
        config=ImaginationEnvConfig(horizon=10, done_threshold=1.1),
    )

    obs = env.reset()
    assert obs.shape == (9,)

    done = False
    steps = 0
    while not done:
        action = np.full(2, 0.5, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        steps += 1
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert 0.0 <= info["done_prob"] <= 1.0

    assert steps == 10

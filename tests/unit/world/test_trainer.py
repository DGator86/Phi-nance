import numpy as np
import torch

from phinance.world.model import RSSMConfig, WorldModelRSSM
from phinance.world.trainer import WorldModelTrainer, build_transition_batch


def test_trainer_step_returns_losses() -> None:
    model = WorldModelRSSM(RSSMConfig(obs_dim=9, action_dim=2, det_dim=32, stoch_dim=8, hidden_dim=32))
    trainer = WorldModelTrainer(model=model)

    n = 16
    obs = np.random.randn(n, 9).astype(np.float32)
    actions = np.random.rand(n, 2).astype(np.float32)
    next_obs = obs + 0.05 * np.random.randn(n, 9).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    dones = np.random.randint(0, 2, size=n).astype(np.float32)
    batch = build_transition_batch(obs, actions, next_obs, rewards, dones)

    losses = trainer.train_step(batch)

    assert set(losses) == {"loss", "recon_loss", "reward_loss", "kl_loss", "done_loss"}
    assert losses["loss"] >= 0.0


def test_save_and_load_round_trip(tmp_path) -> None:
    model = WorldModelRSSM(RSSMConfig(obs_dim=9, action_dim=2))
    trainer = WorldModelTrainer(model=model)
    path = tmp_path / "world_model.pt"
    trainer.save(path)

    loaded = WorldModelTrainer.load(path)

    assert isinstance(loaded, WorldModelRSSM)
    with torch.no_grad():
        assert loaded.config.obs_dim == 9
        assert loaded.config.action_dim == 2

import torch

from phinance.world.model import RSSMConfig, WorldModelRSSM


def test_world_model_forward_shapes() -> None:
    config = RSSMConfig(obs_dim=9, action_dim=2, det_dim=32, stoch_dim=8, hidden_dim=32)
    model = WorldModelRSSM(config)

    batch_size = 4
    prev_state = model.initial_state(batch_size=batch_size)
    obs = torch.randn(batch_size, config.obs_dim)
    action = torch.rand(batch_size, config.action_dim)

    out = model.forward_step(obs=obs, action=action, prev_state=prev_state)

    assert out["recon_obs"].shape == (batch_size, config.obs_dim)
    assert out["reward"].shape == (batch_size,)
    assert out["done_logit"].shape == (batch_size,)
    assert out["prior_mean"].shape == (batch_size, config.stoch_dim)
    assert out["post_std"].shape == (batch_size, config.stoch_dim)


def test_kl_divergence_is_non_negative() -> None:
    config = RSSMConfig(obs_dim=9, action_dim=2)
    model = WorldModelRSSM(config)
    mean_a = torch.zeros(3, config.stoch_dim)
    std_a = torch.ones(3, config.stoch_dim)
    mean_b = torch.randn(3, config.stoch_dim) * 0.1
    std_b = torch.ones(3, config.stoch_dim) * 1.1

    kl = model.kl_divergence(mean_a, std_a, mean_b, std_b)

    assert kl.shape == (3,)
    assert torch.all(kl >= 0.0)

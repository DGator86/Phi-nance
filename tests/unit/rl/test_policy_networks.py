from __future__ import annotations

import numpy as np
import torch

from phinance.rl.policy_networks import CategoricalPolicy, GaussianPolicy


def test_gaussian_policy_supports_all_architectures() -> None:
    obs = torch.rand(4, 9)
    for architecture in ("mlp", "lstm", "transformer"):
        policy = GaussianPolicy(obs_dim=9, architecture=architecture, sequence_length=8)
        mean, std = policy(obs)
        assert mean.shape == (4, 2)
        assert std.shape == (4, 2)
        assert torch.all(mean >= 0.0)
        assert torch.all(mean <= 1.0)


def test_categorical_policy_supports_all_architectures() -> None:
    obs = torch.rand(3, 7)
    for architecture in ("mlp", "lstm", "transformer"):
        policy = CategoricalPolicy(obs_dim=7, n_actions=5, architecture=architecture, sequence_length=6)
        logits = policy(obs)
        assert logits.shape == (3, 5)


def test_policies_accept_prebuilt_sequences() -> None:
    sequence = torch.rand(2, 5, 14)
    discrete = CategoricalPolicy(obs_dim=14, n_actions=4, architecture="lstm", sequence_length=5)
    continuous = GaussianPolicy(obs_dim=14, architecture="transformer", sequence_length=5)

    logits = discrete(sequence)
    mean, std = continuous(sequence)

    assert logits.shape == (2, 4)
    assert mean.shape == (2, 2)
    assert std.shape == (2, 2)
    assert np.isfinite(mean.detach().numpy()).all()

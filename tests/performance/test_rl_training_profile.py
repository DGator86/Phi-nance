"""Performance smoke profile for RL training loops."""

from __future__ import annotations

import copy

from scripts.train_execution_agent import train_with_fallback_loop


def _base_config() -> dict:
    return {
        "environment": {"episode_length": 20},
        "model": {"hidden_size": 64, "architecture": "mlp", "sequence_length": 4},
        "training": {"episodes_smoke": 2, "learning_rate": 1e-3},
    }


def test_rl_training_profile_baseline_vs_optimised(tmp_path):
    base = _base_config()
    baseline_output = tmp_path / "baseline"
    ckpt_base = train_with_fallback_loop(base, baseline_output, optim_cfg={"rl_optimisation": {}})
    assert ckpt_base.exists()

    optim = copy.deepcopy(base)
    optim_cfg = {
        "rl_optimisation": {
            "experience_buffer": {"enabled": True, "size": 2048, "prefetch": True, "batch_size": 32},
            "gpu": {"enabled": False, "mixed_precision": False},
            "numba": {"enabled": True},
            "caching": {"enabled": True},
            "parallel_rollouts": {"enabled": False},
        }
    }
    optim_output = tmp_path / "optim"
    ckpt_optim = train_with_fallback_loop(optim, optim_output, optim_cfg=optim_cfg)
    assert ckpt_optim.exists()

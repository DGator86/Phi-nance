from __future__ import annotations

from pathlib import Path

from phinance.rl.hierarchical.meta_agent import MetaAgent
from phinance.rl.hierarchical.training import load_config, train_with_fallback_loop


def test_hierarchical_fallback_training_saves_checkpoint(tmp_path: Path) -> None:
    config = load_config(Path("configs/meta_agent.yaml"))
    config.setdefault("training", {})["episodes_smoke"] = 2
    config.setdefault("environment", {})["episode_length"] = 8

    checkpoint = train_with_fallback_loop(config=config, output_dir=tmp_path)
    assert checkpoint.exists()

    agent = MetaAgent.load(checkpoint)
    assert agent.n_options >= 3

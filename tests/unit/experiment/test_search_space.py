from __future__ import annotations

import math

from phinance.experiment.search_space import expand_search_space, generate_trial_overrides


def test_grid_generate_trial_overrides() -> None:
    config = {
        "search_space": {
            "params.learning_rate": {"type": "choice", "values": [0.001, 0.01]},
            "params.batch_size": {"type": "int", "low": 32, "high": 33},
        },
        "sweep": {"method": "grid"},
    }

    overrides = generate_trial_overrides(config)
    assert len(overrides) == 4
    assert {trial["params.batch_size"] for trial in overrides} == {32, 33}


def test_random_generate_trial_overrides_respects_ranges() -> None:
    config = {
        "search_space": {
            "params.lr": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
            "params.seed": {"type": "int", "low": 1, "high": 3},
            "data.symbols": {"type": "choice", "values": [["SPY"], ["QQQ"]]},
        },
        "sweep": {"method": "random", "n_trials": 5, "seed": 7},
    }

    overrides = generate_trial_overrides(config)
    assert len(overrides) == 5
    for trial in overrides:
        assert 1e-5 <= trial["params.lr"] <= 1e-3
        assert math.isfinite(trial["params.lr"])
        assert 1 <= trial["params.seed"] <= 3
        assert trial["data.symbols"] in [["SPY"], ["QQQ"]]


def test_expand_search_space_applies_dotted_overrides() -> None:
    config = {
        "name": "demo",
        "target": "phinance.experiment.dummy_targets:gp_discovery_target",
        "params": {"generations": 1},
        "data": {"symbols": ["SPY"]},
        "search_space": {
            "params.generations": {"type": "choice", "values": [2, 3]},
            "data.symbols": {"type": "choice", "values": [["SPY"], ["QQQ"]]},
        },
        "sweep": {"method": "grid"},
    }

    trials = expand_search_space(config)
    assert len(trials) == 4
    assert all("search_space" not in trial for trial in trials)
    assert all("sweep" not in trial for trial in trials)
    values = {(trial["params"]["generations"], tuple(trial["data"]["symbols"])) for trial in trials}
    assert values == {(2, ("SPY",)), (2, ("QQQ",)), (3, ("SPY",)), (3, ("QQQ",))}

"""Path management tests for centralized ``phi.config`` settings."""

from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path

from phi.run_config import RunConfig


def _minimal_config() -> RunConfig:
    return RunConfig(
        dataset_id="paths",
        symbols=["SPY"],
        start_date=date(2022, 1, 1),
        end_date=date(2022, 12, 31),
        indicators={"RSI": {"enabled": True, "params": {"period": 14}}},
        blend_weights={"RSI": 1.0},
    )


def test_settings_paths_defaults(monkeypatch):
    monkeypatch.delenv("DATA_CACHE_DIR", raising=False)
    monkeypatch.delenv("DATA_CACHE_ROOT", raising=False)
    monkeypatch.delenv("RUNS_DIR", raising=False)
    monkeypatch.delenv("LOGS_DIR", raising=False)

    import phi.config as config_module

    config_module = importlib.reload(config_module)
    settings = config_module.Settings()

    assert settings.DATA_CACHE_DIR == Path("./data_cache")
    assert settings.RUNS_DIR == Path("./runs")
    assert settings.LOGS_DIR == Path("./logs")
    assert settings.DATA_CACHE_ROOT == settings.DATA_CACHE_DIR


def test_settings_paths_env_override_and_dir_creation(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    runs_dir = tmp_path / "runs"
    logs_dir = tmp_path / "logs"

    monkeypatch.setenv("DATA_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("LOGS_DIR", str(logs_dir))

    import phi.config as config_module

    config_module = importlib.reload(config_module)
    settings = config_module.Settings()
    settings.create_dirs()

    assert settings.DATA_CACHE_DIR == cache_dir
    assert settings.RUNS_DIR == runs_dir
    assert settings.LOGS_DIR == logs_dir
    assert cache_dir.exists()
    assert runs_dir.exists()
    assert logs_dir.exists()


def test_settings_uses_legacy_data_cache_root(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_CACHE_DIR", raising=False)
    monkeypatch.setenv("DATA_CACHE_ROOT", str(tmp_path / "legacy_cache"))

    import phi.config as config_module

    config_module = importlib.reload(config_module)
    settings = config_module.Settings()

    assert settings.DATA_CACHE_DIR == tmp_path / "legacy_cache"
    assert settings.DATA_CACHE_ROOT == settings.DATA_CACHE_DIR


def test_run_config_save_uses_passed_directory(tmp_path):
    cfg = _minimal_config()
    run_dir = tmp_path / "custom_runs" / "run_001"

    cfg.save(run_dir)

    assert (run_dir / "config.json").exists()

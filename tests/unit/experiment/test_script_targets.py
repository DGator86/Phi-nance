from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from phinance.backtest.models import BacktestResult, Trade
from scripts import run_backtest, run_gp_search, train_meta_agent, train_risk_monitor_agent, train_strategy_rd_agent


def _touch_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ckpt")
    return path


def test_train_strategy_rd_agent_with_and_without_tracker(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_strategy_rd_agent, "_load_config", lambda _: {"training": {"episodes_smoke": 1}})
    monkeypatch.setattr(train_strategy_rd_agent, "load_optimisation_config", lambda _: {"rl_optimisation": {}})

    def fake_train(cfg, output_dir, optim_cfg, tracker=None):
        return _touch_checkpoint(Path(output_dir) / "latest.pt"), {"final_episode_reward": 1.0}

    monkeypatch.setattr(train_strategy_rd_agent, "train_with_fallback_loop", fake_train)

    metrics_no_tracker = train_strategy_rd_agent.train_strategy_rd_agent(
        config="dummy.yaml", optim_config="optim.yaml", output=str(tmp_path / "strategy"), fallback=True, tracker=None
    )
    assert metrics_no_tracker["used_fallback"] == 1.0

    tracker = Mock()
    metrics = train_strategy_rd_agent.train_strategy_rd_agent(
        config="dummy.yaml", optim_config="optim.yaml", output=str(tmp_path / "strategy2"), fallback=True, tracker=tracker
    )
    assert "checkpoint_size_bytes" in metrics
    tracker.log_params.assert_called_once()
    tracker.log_metrics.assert_not_called()
    tracker.log_artifact.assert_called_once()


def test_train_risk_monitor_agent_with_and_without_tracker(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_risk_monitor_agent, "_load_config", lambda _: {"training": {"episodes_smoke": 1}})
    monkeypatch.setattr(train_risk_monitor_agent, "load_optimisation_config", lambda _: {"rl_optimisation": {}})

    def fake_train(cfg, output_dir, optim_cfg, tracker=None):
        return _touch_checkpoint(Path(output_dir) / "latest.pt"), {"final_episode_reward": 2.0}

    monkeypatch.setattr(train_risk_monitor_agent, "train_with_fallback", fake_train)

    assert train_risk_monitor_agent.train_risk_monitor_agent(output=str(tmp_path / "risk"), tracker=None)["used_fallback"] == 1.0
    tracker = Mock()
    train_risk_monitor_agent.train_risk_monitor_agent(output=str(tmp_path / "risk2"), tracker=tracker)
    tracker.log_params.assert_called_once()
    tracker.log_artifact.assert_called_once()


def test_train_meta_agent_with_and_without_tracker(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(train_meta_agent, "load_config", lambda _: {"training": {"episodes_smoke": 1}})
    monkeypatch.setattr(train_meta_agent, "load_optimisation_config", lambda _: {"rl_optimisation": {}})

    def fake_train(cfg, output_dir, optim_cfg, tracker=None):
        return _touch_checkpoint(Path(output_dir) / "latest.pt"), {"final_episode_reward": 3.0}

    monkeypatch.setattr(train_meta_agent, "train_with_fallback_loop", fake_train)

    assert train_meta_agent.train_meta_agent(output=str(tmp_path / "meta"), tracker=None)["used_fallback"] == 1.0
    tracker = Mock()
    train_meta_agent.train_meta_agent(output=str(tmp_path / "meta2"), tracker=tracker)
    tracker.log_params.assert_called_once()
    tracker.log_artifact.assert_called_once()


def test_run_gp_search_tracker(monkeypatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault.json"

    def fake_run(*args, **kwargs):
        vault.write_text("{}", encoding="utf-8")
        return {"best_strategy": {"fitness": 1.23}, "best_strategies": [{}, {}]}

    monkeypatch.setattr(run_gp_search, "run_meta_search", fake_run)

    metrics = run_gp_search.run_gp_search(vault_path=str(vault), tracker=None)
    assert metrics["best_fitness"] == 1.23

    tracker = Mock()
    run_gp_search.run_gp_search(vault_path=str(vault), tracker=tracker)
    tracker.log_params.assert_called_once()
    tracker.log_metrics.assert_called_once()
    tracker.log_artifact.assert_called_once_with(str(vault))


def test_run_backtest_experiment_tracker(monkeypatch, tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1],
            "high": [1.2, 1.2],
            "low": [0.9, 1.0],
            "close": [1.1, 1.15],
            "volume": [100, 120],
        },
        index=pd.date_range("2022-01-01", periods=2, freq="D"),
    )
    monkeypatch.setattr("phinance.data.fetch_and_cache", lambda **_: df)
    monkeypatch.setattr("phinance.optimization.run_phiai_optimization", lambda **kwargs: (kwargs["indicators"], "ok"))

    def fake_backtest(**_):
        return BacktestResult(
            symbol="SPY",
            total_return=0.1,
            cagr=0.08,
            max_drawdown=0.05,
            sharpe=1.2,
            sortino=1.1,
            win_rate=0.6,
            total_trades=1,
            net_pl=100.0,
            trades=[
                Trade(
                    entry_date="2022-01-01",
                    exit_date="2022-01-02",
                    symbol="SPY",
                    entry_price=1.0,
                    exit_price=1.1,
                    quantity=10,
                    pnl=1.0,
                    pnl_pct=0.1,
                    hold_bars=1,
                )
            ],
        )

    monkeypatch.setattr("phinance.backtest.run_backtest", fake_backtest)

    metrics = run_backtest.run_backtest_experiment(log_trades_artifact=False, tracker=None)
    assert metrics["sharpe"] == 1.2

    tracker = Mock()
    artifact = tmp_path / "trades.csv"
    run_backtest.run_backtest_experiment(log_trades_artifact=True, trades_artifact_path=str(artifact), tracker=tracker)
    tracker.log_params.assert_called_once()
    tracker.log_artifact.assert_called_once_with(str(artifact))
    assert artifact.exists()

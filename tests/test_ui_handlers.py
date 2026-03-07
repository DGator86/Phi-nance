from __future__ import annotations

from datetime import date

import pandas as pd

from app_streamlit import ui_handlers


def _base_payload(trading_mode: str = "equities") -> dict:
    return {
        "symbol": "SPY",
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 12, 31),
        "timeframe": "1D",
        "vendor": "alphavantage",
        "initial_capital": 100000.0,
        "trading_mode": trading_mode,
        "indicators": {"RSI": {"enabled": True, "params": {"rsi_period": 14}}},
        "blend_method": "weighted_sum",
        "blend_weights": {"RSI": 1.0},
        "option_type": "call",
        "option_strike": 100.0,
        "option_expiry": date(2023, 12, 31),
        "option_iv": 0.3,
        "option_rate": 0.02,
        "option_qty": 1,
    }


def test_handle_run_backtest_dispatches_equities(monkeypatch):
    calls = {"equity": 0, "options": 0}

    monkeypatch.setattr(ui_handlers, "set_form_errors", lambda errors: None)
    monkeypatch.setattr(ui_handlers, "set_config", lambda config: None)
    monkeypatch.setattr(ui_handlers, "transition_to", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui_handlers, "set_results", lambda results: None)

    class FakeHistory:
        def create_run(self, config):
            return "run_1"

        def save_results(self, run_id, results):
            return None

    monkeypatch.setattr(ui_handlers, "RunHistory", lambda: FakeHistory())

    def fake_load_data(*args, **kwargs):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        return pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1]}, index=idx)

    def fake_equity(**kwargs):
        calls["equity"] += 1
        return {"total_return": 0.1, "portfolio_value": [100000, 110000]}, None

    def fake_options(*args, **kwargs):
        calls["options"] += 1
        return {"total_return": 0.2, "portfolio_value": [100000, 120000]}

    result = ui_handlers.handle_run_backtest(
        _base_payload("equities"),
        load_data_fn=fake_load_data,
        run_equity_fn=fake_equity,
        run_options_fn=fake_options,
    )

    assert result is not None
    assert calls["equity"] == 1
    assert calls["options"] == 0


def test_handle_run_backtest_dispatches_options(monkeypatch):
    calls = {"equity": 0, "options": 0}

    monkeypatch.setattr(ui_handlers, "set_form_errors", lambda errors: None)
    monkeypatch.setattr(ui_handlers, "set_config", lambda config: None)
    monkeypatch.setattr(ui_handlers, "transition_to", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui_handlers, "set_results", lambda results: None)

    class FakeHistory:
        def create_run(self, config):
            return "run_2"

        def save_results(self, run_id, results):
            return None

    monkeypatch.setattr(ui_handlers, "RunHistory", lambda: FakeHistory())

    def fake_load_data(*args, **kwargs):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        return pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1]}, index=idx)

    def fake_equity(**kwargs):
        calls["equity"] += 1
        return {"total_return": 0.1, "portfolio_value": [100000, 110000]}, None

    def fake_options(*args, **kwargs):
        calls["options"] += 1
        return {"total_return": 0.2, "portfolio_value": [100000, 120000]}

    result = ui_handlers.handle_run_backtest(
        _base_payload("options"),
        load_data_fn=fake_load_data,
        run_equity_fn=fake_equity,
        run_options_fn=fake_options,
    )

    assert result is not None
    assert calls["equity"] == 0
    assert calls["options"] == 1

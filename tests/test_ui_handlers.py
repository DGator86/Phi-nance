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


def _patch_state(monkeypatch):
    sink: dict[str, object] = {}
    monkeypatch.setattr(ui_handlers, "set_form_errors", lambda errors: sink.setdefault("form_errors", errors))
    monkeypatch.setattr(ui_handlers, "set_config", lambda config: sink.setdefault("config", config))
    monkeypatch.setattr(ui_handlers, "transition_to", lambda *args, **kwargs: sink.setdefault("transition", (args, kwargs)))
    monkeypatch.setattr(ui_handlers, "set_results", lambda results: sink.setdefault("results", results))
    monkeypatch.setattr(ui_handlers, "set_error", lambda message, debug=None: sink.setdefault("error", (message, debug)))
    return sink


def _fake_history(monkeypatch):
    class FakeHistory:
        def create_run(self, _config):
            return "run_1"

        def save_results(self, _run_id, _results):
            return None

    monkeypatch.setattr(ui_handlers, "RunHistory", lambda: FakeHistory())


def _sample_data():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [1, 1, 1]},
        index=idx,
    )


def test_handle_run_backtest_dispatches_equities(monkeypatch):
    sink = _patch_state(monkeypatch)
    _fake_history(monkeypatch)
    calls = {"equity": 0, "options": 0}

    def fake_equity(**_kwargs):
        calls["equity"] += 1
        return {"total_return": 0.1}, None

    def fake_options(*_args, **_kwargs):
        calls["options"] += 1
        return {"total_return": 0.2}

    result = ui_handlers.handle_run_backtest(
        _base_payload("equities"),
        load_data_fn=lambda *_a, **_k: _sample_data(),
        run_equity_fn=fake_equity,
        run_options_fn=fake_options,
    )

    assert result is not None
    assert calls == {"equity": 1, "options": 0}
    assert sink["results"]["run_id"] == "run_1"


def test_handle_run_backtest_dispatches_options(monkeypatch):
    _patch_state(monkeypatch)
    _fake_history(monkeypatch)
    calls = {"equity": 0, "options": 0}

    def fake_equity(**_kwargs):
        calls["equity"] += 1
        return {"total_return": 0.1}, None

    def fake_options(*_args, **_kwargs):
        calls["options"] += 1
        return {"total_return": 0.2}

    result = ui_handlers.handle_run_backtest(
        _base_payload("options"),
        load_data_fn=lambda *_a, **_k: _sample_data(),
        run_equity_fn=fake_equity,
        run_options_fn=fake_options,
    )

    assert result is not None
    assert calls == {"equity": 0, "options": 1}


def test_handle_run_backtest_validation_failure_sets_form_errors(monkeypatch):
    sink = _patch_state(monkeypatch)

    payload = _base_payload()
    payload["symbol"] = ""

    result = ui_handlers.handle_run_backtest(payload)

    assert result is None
    assert "Symbol is required." in sink["form_errors"]


def test_handle_run_backtest_exception_sets_error(monkeypatch):
    sink = _patch_state(monkeypatch)
    _fake_history(monkeypatch)

    result = ui_handlers.handle_run_backtest(
        _base_payload(),
        load_data_fn=lambda *_a, **_k: _sample_data(),
        run_equity_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert result is None
    assert "Backtest failed" in sink["error"][0]

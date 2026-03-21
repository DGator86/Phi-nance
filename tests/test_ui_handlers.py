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
    monkeypatch.setattr(
        ui_handlers,
        "enrich_run_config_options_from_uw",
        lambda cfg, data: (cfg, {"flow_summary": {"alerts": 0}, "greeks_from_chain": {"delta": 0.5}}),
    )
    calls = {"equity": 0, "options": 0}

    def fake_equity(**_kwargs):
        calls["equity"] += 1
        return {"total_return": 0.1}, None

    def fake_options(*_args, **_kwargs):
        calls["options"] += 1
        return {"total_return": 0.2}

    p = _base_payload("options")
    p["vendor"] = "unusual_whales"
    result = ui_handlers.handle_run_backtest(
        p,
        load_data_fn=lambda *_a, **_k: _sample_data(),
        run_equity_fn=fake_equity,
        run_options_fn=fake_options,
    )

    assert result is not None
    assert calls == {"equity": 0, "options": 1}
    assert result.get("unusual_whales_context", {}).get("flow_summary", {}).get("alerts") == 0


def test_options_mode_requires_unusual_whales_vendor(monkeypatch):
    sink = _patch_state(monkeypatch)
    p = _base_payload("options")
    p["vendor"] = "yfinance"
    result = ui_handlers.handle_run_backtest(p)
    assert result is None
    assert any("unusual_whales" in m.lower() for m in sink["form_errors"])


def test_handle_run_backtest_validation_failure_sets_form_errors(monkeypatch):
    sink = _patch_state(monkeypatch)

    payload = _base_payload()
    payload["symbol"] = ""

    result = ui_handlers.handle_run_backtest(payload)

    assert result is None
    assert any("symbol" in msg.lower() for msg in sink["form_errors"])


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


def test_build_run_config_auto_assigns_equal_blend_weights():
    payload = _base_payload("equities")
    payload["blend_weights"] = {}
    payload["indicators"] = {
        "RSI": {"enabled": True, "params": {"rsi_period": 14}},
        "MACD": {"enabled": True, "params": {}},
    }

    cfg = ui_handlers.build_run_config(payload)

    assert sum(cfg.blend_weights.values()) == 1.0
    assert set(cfg.blend_weights) == {"RSI", "MACD"}


def test_handle_load_run_sets_error_when_missing(monkeypatch):
    sink = _patch_state(monkeypatch)

    class MissingHistory:
        def load_config(self, _run_id):
            return None

        def load_results(self, _run_id):
            return None

    monkeypatch.setattr(ui_handlers, "RunHistory", lambda: MissingHistory())

    result = ui_handlers.handle_load_run("does-not-exist")

    assert result is None
    assert "could not be loaded" in sink["error"][0]


def test_handle_load_run_sets_config_and_results(monkeypatch):
    sink = _patch_state(monkeypatch)
    cfg = ui_handlers.build_run_config(_base_payload())

    class PresentHistory:
        def load_config(self, _run_id):
            return cfg

        def load_results(self, _run_id):
            return {"total_return": 0.15}

    monkeypatch.setattr(ui_handlers, "RunHistory", lambda: PresentHistory())

    result = ui_handlers.handle_load_run("run_2")

    assert result["run_id"] == "run_2"
    assert sink["config"]["symbols"] == ["SPY"]
    assert sink["results"]["total_return"] == 0.15


def test_validate_config_payload_rejects_invalid_symbol_and_numbers():
    payload = _base_payload()
    payload["symbol"] = "../spy"
    payload["initial_capital"] = -1

    errors = ui_handlers.validate_config_payload(payload)

    assert any("Ticker symbol" in msg for msg in errors)
    assert any("Initial capital" in msg for msg in errors)


def test_handle_load_run_rejects_unsafe_run_id(monkeypatch):
    sink = _patch_state(monkeypatch)

    result = ui_handlers.handle_load_run("../bad")

    assert result is None
    assert "Run ID" in sink["error"][0]



def test_handle_train_regime_detector_unknown_method_warns_and_falls_back(monkeypatch):
    class _SessionState:
        """Minimal Streamlit-like session_state (attribute + key assignment)."""

        def __init__(self) -> None:
            object.__setattr__(self, "_data", {})

        def __getitem__(self, key: str):
            return self._data[key]

        def __setitem__(self, key: str, value) -> None:
            self._data[key] = value

        def __getattr__(self, name: str):
            if name == "_data":
                raise AttributeError(name)
            return self._data[name]

        def __setattr__(self, name: str, value) -> None:
            if name == "_data":
                object.__setattr__(self, name, value)
            else:
                object.__getattribute__(self, "_data")[name] = value

    class FakeSt:
        session_state = _SessionState()

    captured: dict[str, object] = {}

    class FakeDetector:
        def predict(self, data):
            return pd.Series(["state_0"] * len(data), index=data.index)

    monkeypatch.setattr(ui_handlers, "st", FakeSt)
    monkeypatch.setattr(ui_handlers.logger, "warning", lambda *args, **kwargs: captured.setdefault("warned", True))

    def fake_train(data, method, n_regimes, window, save):
        captured["method"] = method
        return FakeDetector(), None

    monkeypatch.setattr(ui_handlers, "train_regime_detector", fake_train)

    payload = _base_payload()
    payload["regime_method"] = "UnknownLabel"
    payload["regime_n_states"] = 3
    payload["regime_window"] = 20

    detector, regimes, path = ui_handlers.handle_train_regime_detector(
        payload,
        load_data_fn=lambda *_a, **_k: _sample_data(),
    )

    assert detector is not None
    assert not regimes.empty
    assert path is None
    assert captured["method"] == "kmeans"
    assert captured.get("warned") is True


def test_handle_run_backtest_passes_regime_label_map(monkeypatch):
    sink = _patch_state(monkeypatch)
    _fake_history(monkeypatch)

    class FakeDetector:
        def predict(self, data):
            return pd.Series(["state_0"] * len(data), index=data.index)

    monkeypatch.setattr(ui_handlers, "load_detector_from_payload", lambda _p: FakeDetector())

    seen: dict[str, object] = {}

    def fake_equity(**kwargs):
        seen["regime_label_map"] = kwargs.get("regime_label_map")
        return {"total_return": 0.1}, None

    payload = _base_payload("equities")
    payload["regime_enabled"] = True
    payload["regime_label_map"] = {"state_0": "bull"}

    result = ui_handlers.handle_run_backtest(
        payload,
        load_data_fn=lambda *_a, **_k: _sample_data(),
        run_equity_fn=fake_equity,
        run_options_fn=lambda *_a, **_k: {"total_return": 0.0},
    )

    assert result is not None
    assert seen["regime_label_map"] == {"state_0": "bull"}
    assert sink["results"]["run_id"] == "run_1"


def test_handle_run_backtest_multi_symbol_uses_portfolio_engine(monkeypatch):
    sink = _patch_state(monkeypatch)
    _fake_history(monkeypatch)

    called = {"portfolio": 0, "equity": 0}

    def fake_portfolio(**_kwargs):
        called["portfolio"] += 1
        return {"total_return": 0.3, "portfolio_value": [100000, 101000], "final_weights": {"SPY": 0.5, "QQQ": 0.5}}

    def fake_equity(**_kwargs):
        called["equity"] += 1
        return {"total_return": 0.1}, None

    monkeypatch.setattr(ui_handlers, "run_portfolio_backtest", fake_portfolio)

    payload = _base_payload("equities")
    payload["symbols"] = ["SPY", "QQQ"]
    payload["allocation_strategy"] = "equal_weight"
    payload["allocation_params"] = {}
    payload["rebalance_frequency"] = "W"
    payload["rebalance_threshold"] = None

    result = ui_handlers.handle_run_backtest(
        payload,
        load_data_fn=lambda *_a, **_k: _sample_data(),
        run_equity_fn=fake_equity,
        run_options_fn=lambda *_a, **_k: {"total_return": 0.0},
    )

    assert result is not None
    assert called == {"portfolio": 1, "equity": 0}
    assert sink["results"]["run_id"] == "run_1"

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app_streamlit import ui_handlers
from phi.backtest.direct import run_direct_backtest
from phi.exceptions import BacktestError


def _sample_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.Series(range(100, 130), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )


def test_run_direct_backtest_regime_boosts_accepts_detector_labels() -> None:
    data = _sample_ohlcv()
    indicators = {"Buy & Hold": {"enabled": True, "params": {}}}
    weights = {"Buy & Hold": 1.0}
    regimes = pd.Series(["state_0"] * len(data), index=data.index)

    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators=indicators,
        blend_weights=weights,
        blend_method="regime_weighted",
        regime_series=regimes,
        regime_label_map={"state_0": "bull"},
        regime_boosts={"bull": {"Buy & Hold": 1.2}},
    )

    assert "total_return" in results
    assert len(results["portfolio_value"]) > 1


def test_run_direct_backtest_regime_uses_detector_on_the_fly() -> None:
    class FakeDetector:
        def predict(self, ohlcv: pd.DataFrame) -> pd.Series:
            return pd.Series(["state_0"] * len(ohlcv), index=ohlcv.index)

    data = _sample_ohlcv()
    results, _ = run_direct_backtest(
        ohlcv=data,
        symbol="SPY",
        indicators={"Buy & Hold": {"enabled": True, "params": {}}},
        blend_weights={"Buy & Hold": 1.0},
        blend_method="regime_weighted",
        regime_detector=FakeDetector(),
        regime_boosts={"state_0": {"Buy & Hold": 1.0}},
    )
    assert len(results["portfolio_value"]) == len(data) + 1


def test_run_direct_backtest_regime_requires_series_or_detector() -> None:
    with pytest.raises(BacktestError):
        run_direct_backtest(
            ohlcv=_sample_ohlcv(),
            symbol="SPY",
            indicators={"Buy & Hold": {"enabled": True, "params": {}}},
            blend_weights={"Buy & Hold": 1.0},
            blend_method="regime_weighted",
        )


def test_build_regime_boosts_from_payload_maps_friendly_labels() -> None:
    payload = {
        "regime_label_map": {"state_0": "Bull", "state_1": "Bear"},
        "regime_boost_matrix": {
            "state_0": {"RSI": 1.2, "MACD": 0.9},
            "state_1": {"RSI": 0.8, "MACD": 1.1},
        },
    }

    boosts = ui_handlers.build_regime_boosts_from_payload(payload)

    assert boosts["Bull"]["RSI"] == 1.2
    assert boosts["Bear"]["MACD"] == 1.1


def test_handle_run_backtest_regime_aware_passes_boosts(monkeypatch) -> None:
    class FakeSt:
        session_state = {}

    monkeypatch.setattr(ui_handlers, "st", FakeSt)

    sink: dict[str, object] = {}
    monkeypatch.setattr(ui_handlers, "set_form_errors", lambda errors: sink.setdefault("form_errors", errors))
    monkeypatch.setattr(ui_handlers, "set_config", lambda config: sink.setdefault("config", config))
    monkeypatch.setattr(ui_handlers, "transition_to", lambda *args, **kwargs: None)
    monkeypatch.setattr(ui_handlers, "set_results", lambda results: sink.setdefault("results", results))
    monkeypatch.setattr(ui_handlers, "set_error", lambda message, debug=None: sink.setdefault("error", (message, debug)))

    class FakeHistory:
        def create_run(self, _config):
            return "run_aware_1"

        def save_results(self, _run_id, _results):
            return None

    monkeypatch.setattr(ui_handlers, "RunHistory", lambda: FakeHistory())

    class FakeDetector:
        def predict(self, data):
            return pd.Series(["state_0"] * len(data), index=data.index)

    monkeypatch.setattr(ui_handlers, "load_detector_from_payload", lambda _payload: FakeDetector())

    observed: dict[str, object] = {}

    def fake_equity(**kwargs):
        observed["regime_boosts"] = kwargs.get("regime_boosts")
        observed["regime_label_map"] = kwargs.get("regime_label_map")
        return {"total_return": 0.1}, None

    payload = {
        "symbol": "SPY",
        "start_date": date(2023, 1, 1),
        "end_date": date(2023, 12, 31),
        "timeframe": "1D",
        "vendor": "alphavantage",
        "initial_capital": 100000.0,
        "trading_mode": "equities",
        "indicators": {"RSI": {"enabled": True, "params": {"rsi_period": 14}}},
        "blend_method": "regime_weighted",
        "blend_weights": {"RSI": 1.0},
        "regime_enabled": True,
        "regime_detect_on_the_fly": True,
        "regime_use_precomputed": False,
        "regime_selected_model_path": "runs/regime_models/fake.pkl",
        "regime_label_map": {"state_0": "Bull"},
        "regime_boost_matrix": {"state_0": {"RSI": 1.2}},
    }

    result = ui_handlers.handle_run_backtest(
        payload,
        load_data_fn=lambda *_a, **_k: _sample_ohlcv(),
        run_equity_fn=fake_equity,
    )

    assert result is not None
    assert observed["regime_label_map"] == {"state_0": "Bull"}
    assert observed["regime_boosts"] == {"Bull": {"RSI": 1.2}}

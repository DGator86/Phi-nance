from __future__ import annotations

from app_streamlit import state


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _patch_session_state(monkeypatch):
    ss = SessionState()
    monkeypatch.setattr(state.st, "session_state", ss)
    return ss


def test_init_session_state_populates_defaults(monkeypatch):
    ss = _patch_session_state(monkeypatch)

    state.init_session_state()

    for key in state.DEFAULT_STATE:
        assert key in ss


def test_set_results_clears_errors_and_transitions(monkeypatch):
    ss = _patch_session_state(monkeypatch)
    state.init_session_state()
    ss["error"] = "old"
    ss["error_debug"] = "trace"

    state.set_results({"total_return": 0.1})

    assert ss["results"] == {"total_return": 0.1}
    assert ss["error"] is None
    assert ss["error_debug"] is None
    assert ss["app_state"] == state.AppState.RESULTS


def test_reset_state_clears_config_inputs(monkeypatch):
    ss = _patch_session_state(monkeypatch)
    state.init_session_state()
    ss["symbol"] = "SPY"
    ss["option_qty"] = 2

    state.reset_state()

    assert "symbol" not in ss
    assert "option_qty" not in ss
    assert ss["app_state"] == state.AppState.IDLE

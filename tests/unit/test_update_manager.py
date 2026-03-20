from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from phi.utils.updater import UpdateManager


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_should_check_respects_weekly_cadence(tmp_path: Path):
    state = tmp_path / "state.json"
    req = tmp_path / "requirements.txt"
    req.write_text("lumibot==4.4.52\n", encoding="utf-8")

    mgr = UpdateManager(requirements_path=req, state_file=state, cadence="weekly")
    assert mgr.should_check() is True

    mgr._save_last_check(datetime.now(timezone.utc))
    assert mgr.should_check() is False

    mgr._save_last_check(datetime.now(timezone.utc) - timedelta(days=8))
    assert mgr.should_check() is True


def test_check_all_sets_session_state_with_updates(tmp_path: Path, monkeypatch):
    req = tmp_path / "requirements.txt"
    req.write_text("lumibot==4.4.51\nheadroom-ai==0.3.6\n", encoding="utf-8")

    def fake_get(url, timeout=0, headers=None):
        if "pypi.org/pypi/lumibot" in url:
            return DummyResponse(payload={"info": {"version": "4.4.52"}})
        if "pypi.org/pypi/headroom-ai" in url:
            return DummyResponse(payload={"info": {"version": "0.3.7"}})
        if "je-suis-tm/quant-trading" in url:
            return DummyResponse(payload={"sha": "abc1234567890"})
        if "tzachbon/smart-ralph" in url:
            return DummyResponse(payload={"sha": "def1234567890"})
        raise AssertionError(url)

    monkeypatch.setattr("phi.utils.updater.requests.get", fake_get)

    session = {}
    mgr = UpdateManager(requirements_path=req, state_file=tmp_path / "state.json", session_state=session, cadence="startup")
    statuses = mgr.check_all(force=True)

    assert len(statuses) == 4
    assert len(session["tool_updates_available"]) == 2
    names = {item.display_name for item in session["tool_updates_available"]}
    assert names == {"Lumibot", "Headroom"}

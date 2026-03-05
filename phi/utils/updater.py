"""Update checks for external Phi-nance dependencies."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class ToolStatus:
    """Represents current and latest version info for a tracked tool."""

    key: str
    display_name: str
    current_version: str | None
    latest_version: str | None
    update_available: bool
    source: str
    details: str = ""


class UpdateManager:
    """Checks external tools for updates and optionally performs upgrades."""

    _CADENCE_WINDOWS = {
        "startup": timedelta(seconds=0),
        "daily": timedelta(days=1),
        "weekly": timedelta(days=7),
        "monthly": timedelta(days=30),
    }

    def __init__(
        self,
        *,
        requirements_path: str | Path = "requirements.txt",
        cadence: str | None = None,
        state_file: str | Path | None = None,
        session_state: dict[str, Any] | None = None,
        timeout_s: int = 8,
    ) -> None:
        self.requirements_path = Path(requirements_path)
        self.cadence = (cadence or os.getenv("UPDATE_CHECK_CADENCE", "weekly")).strip().lower()
        self.state_file = Path(state_file or Path.home() / ".phinance" / "updater_last_check.json")
        self.session_state = session_state
        self.timeout_s = timeout_s

        if self.cadence not in {*self._CADENCE_WINDOWS.keys(), "never"}:
            self.cadence = "weekly"

    def should_check(self) -> bool:
        """Return True if cadence allows checking now."""
        if self.cadence == "never":
            return False

        last_check = self._load_last_check()
        if last_check is None:
            return True

        return datetime.now(timezone.utc) - last_check >= self._CADENCE_WINDOWS[self.cadence]

    def check_all(self, *, force: bool = False) -> list[ToolStatus]:
        """Check all tracked tools for updates."""
        if not force and not self.should_check():
            return []

        statuses = [
            self.check_tool("headroom"),
            self.check_tool("lumibot"),
            self.check_tool("quant-trading"),
            self.check_tool("smart-ralph"),
        ]
        self._save_last_check(datetime.now(timezone.utc))
        self.notify_user(statuses)
        return statuses

    def check_tool(self, tool_key: str) -> ToolStatus:
        """Check one tool using PyPI or GitHub metadata."""
        tool_key = tool_key.lower()
        if tool_key == "headroom":
            return self._check_pypi(tool_key, "Headroom", "headroom-ai")
        if tool_key == "lumibot":
            return self._check_pypi(tool_key, "Lumibot", "lumibot")
        if tool_key == "quant-trading":
            return self._check_github(tool_key, "quant-trading", "je-suis-tm", "quant-trading")
        if tool_key == "smart-ralph":
            return self._check_github(tool_key, "Smart Ralph", "tzachbon", "smart-ralph")
        raise ValueError(f"Unsupported tool: {tool_key}")

    def notify_user(self, statuses: list[ToolStatus]) -> None:
        """Expose update metadata to Streamlit session state."""
        if self.session_state is None:
            return
        updates = [s for s in statuses if s.update_available]
        self.session_state["tool_updates"] = statuses
        self.session_state["tool_updates_available"] = updates
        self.session_state["tool_updates_checked_at"] = datetime.now(timezone.utc).isoformat()

    def perform_update(self, tool_key: str, version: str | None = None) -> str:
        """Update requirements and run pip install for the selected tool."""
        tool_key = tool_key.lower()
        if tool_key in {"headroom", "lumibot"}:
            package_name = "headroom-ai" if tool_key == "headroom" else "lumibot"
            spec = f"{package_name}=={version}" if version else package_name
            self._update_requirements_package(package_name, version)
            result = subprocess.run(
                ["python", "-m", "pip", "install", "--upgrade", spec],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"Failed to update {package_name}")
            return result.stdout.strip() or f"Updated {package_name}"

        if tool_key in {"quant-trading", "smart-ralph"}:
            repo_dir = Path("external") / tool_key
            if not repo_dir.exists():
                raise FileNotFoundError(
                    f"{repo_dir} not found. Clone the repository to enable git pull updates."
                )
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "pull", "--ff-only"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"Failed to update {tool_key}")
            return result.stdout.strip() or f"Updated {tool_key}"

        raise ValueError(f"Unsupported tool: {tool_key}")

    def _check_pypi(self, key: str, display_name: str, package_name: str) -> ToolStatus:
        current = self._read_required_version(package_name)
        try:
            response = requests.get(
                f"https://pypi.org/pypi/{package_name}/json", timeout=self.timeout_s
            )
            response.raise_for_status()
            latest = response.json().get("info", {}).get("version")
        except Exception as exc:  # noqa: BLE001 - we surface error details to UI.
            return ToolStatus(
                key=key,
                display_name=display_name,
                current_version=current,
                latest_version=None,
                update_available=False,
                source="pypi",
                details=f"Unable to fetch latest version: {exc}",
            )

        return ToolStatus(
            key=key,
            display_name=display_name,
            current_version=current,
            latest_version=latest,
            update_available=bool(current and latest and current != latest),
            source="pypi",
        )

    def _check_github(self, key: str, display_name: str, owner: str, repo: str) -> ToolStatus:
        current = self._read_local_repo_sha(key)
        try:
            response = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/commits/main",
                timeout=self.timeout_s,
                headers={"Accept": "application/vnd.github+json"},
            )
            if response.status_code == 404:
                response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/commits/master",
                    timeout=self.timeout_s,
                    headers={"Accept": "application/vnd.github+json"},
                )
            response.raise_for_status()
            latest = (response.json().get("sha") or "")[:12] or None
        except Exception as exc:  # noqa: BLE001
            return ToolStatus(
                key=key,
                display_name=display_name,
                current_version=current,
                latest_version=None,
                update_available=False,
                source="github",
                details=f"Unable to fetch latest commit: {exc}",
            )

        return ToolStatus(
            key=key,
            display_name=display_name,
            current_version=current,
            latest_version=latest,
            update_available=bool(current and latest and current != latest),
            source="github",
            details=("Tracked via latest default-branch commit SHA."),
        )

    def _read_required_version(self, package_name: str) -> str | None:
        if not self.requirements_path.exists():
            return None
        for raw_line in self.requirements_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(package_name):
                for sep in ("==", ">=", "~=", "<="):
                    if sep in line:
                        return line.split(sep, 1)[1].strip()
                return None
        return None

    def _update_requirements_package(self, package_name: str, version: str | None) -> None:
        if not self.requirements_path.exists():
            raise FileNotFoundError(f"{self.requirements_path} was not found")

        lines = self.requirements_path.read_text(encoding="utf-8").splitlines()
        replacement = f"{package_name}=={version}" if version else package_name
        updated = False

        for idx, raw_line in enumerate(lines):
            line = raw_line.split("#", 1)[0].strip()
            if line.startswith(package_name):
                lines[idx] = replacement
                updated = True
                break

        if not updated:
            lines.append(replacement)

        self.requirements_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _read_local_repo_sha(self, tool_key: str) -> str | None:
        repo_dir = Path("external") / tool_key
        if not repo_dir.exists():
            return None
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _load_last_check(self) -> datetime | None:
        if not self.state_file.exists():
            return None
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            raw = payload.get("last_check")
            if not raw:
                return None
            return datetime.fromisoformat(raw)
        except Exception:  # noqa: BLE001
            return None

    def _save_last_check(self, when: datetime) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_check": when.astimezone(timezone.utc).isoformat()}
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

"""
phinance.config — Configuration management.

Sub-modules
-----------
  run_config — RunConfig Pydantic-like model + RunHistory
  settings   — Global application settings (API keys, paths)
  schema     — Validation helpers

Public API
----------
    from phinance.config import RunConfig, Settings
"""

from phinance.config.run_config import RunConfig
from phinance.config.settings import Settings, get_settings

__all__ = ["RunConfig", "Settings", "get_settings"]

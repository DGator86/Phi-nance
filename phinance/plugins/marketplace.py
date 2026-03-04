"""
phinance.plugins.marketplace
================================

Plugin Marketplace — versioned, discoverable plugin registry with
semantic versioning, dependency resolution, conflict detection, and
a human-readable plugin catalogue.

Architecture
------------
The marketplace extends the base PluginRegistry with:

  * **Versioning**        — each plugin carries a SemVer string
  * **Manifest**          — PluginManifest dataclass (replaces raw dict metadata)
  * **Dependency check**  — warns if required Python packages are missing
  * **Conflict detection**— flags duplicate registrations across versions
  * **Catalogue**         — list / search / filter plugins
  * **SDK scaffold**      — generate boilerplate for a new plugin

Public API
----------
  PluginManifest       — rich metadata for a marketplace plugin
  PluginEntry          — versioned registry entry
  MarketplaceRegistry  — extended registry with versioning & discovery
  get_marketplace      — module-level singleton accessor
  reset_marketplace    — reset the singleton (test helper)
  scaffold_plugin      — generate a plugin skeleton string
"""

from __future__ import annotations

import importlib
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from phinance.strategies.base import BaseIndicator
from phinance.plugins.registry import PluginRegistry, get_registry
from phinance.utils.logging import get_logger

logger = get_logger(__name__)

# Simple SemVer regex (major.minor.patch)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PluginManifest:
    """
    Rich metadata for a marketplace plugin.

    Attributes
    ----------
    name          : str  — unique plugin name (e.g. "My RSI Fork")
    version       : str  — SemVer string (e.g. "1.2.0")
    author        : str
    description   : str
    plugin_type   : str  — "indicator" | "vendor" | "blender" | "other"
    requires      : list — Python package names required at runtime
    tags          : list — free-form tags for search
    homepage      : str  — URL
    license       : str  — SPDX identifier (e.g. "MIT")
    plugin_id     : str  — auto-generated UUID
    """

    name:        str = ""
    version:     str = "0.1.0"
    author:      str = "Anonymous"
    description: str = ""
    plugin_type: str = "indicator"
    requires:    List[str] = field(default_factory=list)
    tags:        List[str] = field(default_factory=list)
    homepage:    str = ""
    license:     str = "MIT"
    plugin_id:   str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def is_valid_version(self) -> bool:
        return bool(_SEMVER_RE.match(self.version))

    def missing_dependencies(self) -> List[str]:
        """Return list of required packages that cannot be imported."""
        missing = []
        for pkg in self.requires:
            try:
                importlib.import_module(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        return missing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "version":     self.version,
            "author":      self.author,
            "description": self.description,
            "plugin_type": self.plugin_type,
            "requires":    self.requires,
            "tags":        self.tags,
            "homepage":    self.homepage,
            "license":     self.license,
            "plugin_id":   self.plugin_id,
        }

    def __repr__(self) -> str:
        return f"PluginManifest(name={self.name!r}, version={self.version}, type={self.plugin_type})"


@dataclass
class PluginEntry:
    """
    One versioned entry in the MarketplaceRegistry.

    Attributes
    ----------
    manifest    : PluginManifest
    plugin_cls  : Type — the plugin class (indicator / vendor / etc.)
    installed   : bool — True after successful registration in base registry
    """

    manifest:   PluginManifest
    plugin_cls: Any
    installed:  bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.manifest.to_dict(),
            "installed": self.installed,
        }

    def __repr__(self) -> str:
        return f"PluginEntry({self.manifest.name!r} v{self.manifest.version}, installed={self.installed})"


# ─────────────────────────────────────────────────────────────────────────────
# MarketplaceRegistry
# ─────────────────────────────────────────────────────────────────────────────


class MarketplaceRegistry:
    """
    Versioned, discoverable plugin marketplace.

    Usage
    -----
    ::

        from phinance.plugins.marketplace import get_marketplace, PluginManifest

        mp = get_marketplace()

        @mp.publish(PluginManifest(name="My RSI", version="1.0.0", author="Alice"))
        class MyRSI(BaseIndicator):
            ...

        entry = mp.get("My RSI")
        results = mp.search(tags=["momentum"])
    """

    def __init__(self) -> None:
        # name → list of PluginEntry (all versions, sorted by version ascending)
        self._entries: Dict[str, List[PluginEntry]] = {}
        self._base_registry = get_registry()

    # ── registration ─────────────────────────────────────────────────────────

    def publish(
        self,
        manifest: PluginManifest,
        overwrite: bool = False,
    ):
        """
        Decorator: publish a plugin class to the marketplace.

        Usage::

            @mp.publish(PluginManifest(name="My RSI", version="1.0.0"))
            class MyRSI(BaseIndicator):
                ...
        """
        def decorator(cls):
            self._register_entry(manifest, cls, overwrite=overwrite)
            return cls
        return decorator

    def register(
        self,
        plugin_cls: Any,
        manifest: PluginManifest,
        overwrite: bool = False,
    ) -> PluginEntry:
        """Register a plugin class with a manifest; returns the PluginEntry."""
        return self._register_entry(manifest, plugin_cls, overwrite=overwrite)

    def _register_entry(
        self,
        manifest: PluginManifest,
        plugin_cls: Any,
        overwrite: bool = False,
    ) -> PluginEntry:
        name = manifest.name
        if not name:
            raise ValueError("PluginManifest.name must not be empty")
        if not manifest.is_valid_version():
            raise ValueError(
                f"Invalid version {manifest.version!r} for plugin {name!r}. "
                "Use SemVer (major.minor.patch)."
            )

        # Check for duplicate version
        existing = self._entries.get(name, [])
        for e in existing:
            if e.manifest.version == manifest.version and not overwrite:
                raise ValueError(
                    f"Plugin {name!r} v{manifest.version} already registered. "
                    "Use overwrite=True to replace."
                )

        # Warn about missing deps
        missing = manifest.missing_dependencies()
        if missing:
            logger.warning(
                "Plugin %r v%s has missing dependencies: %s",
                name, manifest.version, missing,
            )

        # Install into base registry if it's an indicator
        installed = False
        if manifest.plugin_type == "indicator":
            try:
                self._base_registry.register_indicator(
                    name=name,
                    indicator=plugin_cls,
                    metadata=manifest.to_dict(),
                    overwrite=overwrite,
                )
                installed = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not install %r into base registry: %s", name, exc)
        elif manifest.plugin_type == "vendor":
            try:
                self._base_registry.register_vendor(
                    name=name,
                    vendor=plugin_cls,
                    metadata=manifest.to_dict(),
                    overwrite=overwrite,
                )
                installed = True
            except Exception:  # noqa: BLE001
                pass

        entry = PluginEntry(manifest=manifest, plugin_cls=plugin_cls, installed=installed)

        # Replace old version if overwrite; otherwise append
        if overwrite:
            self._entries[name] = [
                e for e in existing if e.manifest.version != manifest.version
            ]
        else:
            if name not in self._entries:
                self._entries[name] = []

        self._entries[name].append(entry)
        # Keep sorted by version (lexicographic is fine for simple SemVer)
        self._entries[name].sort(key=lambda e: e.manifest.version)

        logger.info("Published plugin %r v%s (type=%s)", name, manifest.version, manifest.plugin_type)
        return entry

    # ── discovery ────────────────────────────────────────────────────────────

    def get(self, name: str, version: Optional[str] = None) -> Optional[PluginEntry]:
        """
        Get a plugin entry by name (and optionally version).
        If version is None, returns the latest version.
        """
        entries = self._entries.get(name, [])
        if not entries:
            return None
        if version is None:
            return entries[-1]  # latest
        for e in entries:
            if e.manifest.version == version:
                return e
        return None

    def list_all(self) -> List[PluginEntry]:
        """Return all plugin entries (all names, latest version each)."""
        return [entries[-1] for entries in self._entries.values() if entries]

    def list_versions(self, name: str) -> List[str]:
        """Return all registered versions for a plugin name."""
        return [e.manifest.version for e in self._entries.get(name, [])]

    def search(
        self,
        query: str = "",
        plugin_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[PluginEntry]:
        """
        Search the marketplace.

        Parameters
        ----------
        query       : str  — substring match on name or description
        plugin_type : str  — filter by plugin_type
        tags        : list — filter by tags (ANY match)

        Returns
        -------
        list[PluginEntry]
        """
        results = self.list_all()

        if query:
            q = query.lower()
            results = [
                e for e in results
                if q in e.manifest.name.lower() or q in e.manifest.description.lower()
            ]
        if plugin_type:
            results = [e for e in results if e.manifest.plugin_type == plugin_type]
        if tags:
            tag_set = set(t.lower() for t in tags)
            results = [
                e for e in results
                if tag_set & {t.lower() for t in e.manifest.tags}
            ]

        return results

    def catalogue(self) -> List[Dict[str, Any]]:
        """Return a list of dicts for all plugins (latest versions)."""
        return [e.to_dict() for e in self.list_all()]

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MarketplaceRegistry(plugins={len(self)})"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_marketplace: Optional[MarketplaceRegistry] = None


def get_marketplace() -> MarketplaceRegistry:
    """Return the module-level MarketplaceRegistry singleton."""
    global _marketplace
    if _marketplace is None:
        _marketplace = MarketplaceRegistry()
    return _marketplace


def reset_marketplace() -> None:
    """Reset the singleton (use in tests only)."""
    global _marketplace
    _marketplace = None


# ─────────────────────────────────────────────────────────────────────────────
# Plugin SDK scaffold
# ─────────────────────────────────────────────────────────────────────────────


def scaffold_plugin(
    name: str,
    author: str = "Your Name",
    version: str = "0.1.0",
    plugin_type: str = "indicator",
    description: str = "",
) -> str:
    """
    Generate a plugin skeleton string.

    Parameters
    ----------
    name        : str — plugin display name
    author      : str
    version     : str — SemVer
    plugin_type : str — "indicator" | "vendor"
    description : str

    Returns
    -------
    str  — Python source code for the plugin skeleton
    """
    cls_name = "".join(w.capitalize() for w in name.split()) + (
        "Indicator" if plugin_type == "indicator" else "Vendor"
    )
    desc = description or f"{name} — a {plugin_type} plugin for Phi-nance"

    if plugin_type == "indicator":
        code = textwrap.dedent(f'''\
            """
            {name} — Phi-nance indicator plugin
            Author  : {author}
            Version : {version}
            """

            from __future__ import annotations
            import pandas as pd
            from phinance.strategies.base import BaseIndicator
            from phinance.plugins.marketplace import get_marketplace, PluginManifest

            _mp = get_marketplace()

            @_mp.publish(PluginManifest(
                name="{name}",
                version="{version}",
                author="{author}",
                description="{desc}",
                plugin_type="indicator",
                tags=[],
            ))
            class {cls_name}(BaseIndicator):
                _NAME = "{name}"

                def __init__(self, period: int = 14) -> None:
                    super().__init__()
                    self.period = period

                @property
                def name(self) -> str:
                    return self._NAME

                @property
                def param_grid(self) -> dict:
                    return {{"period": [7, 14, 21]}}

                def compute(self, ohlcv: pd.DataFrame, params: dict | None = None) -> pd.Series:
                    p = (params or {{}}).get("period", self.period)
                    close = ohlcv["close"]
                    # TODO: implement your signal here (return values in [-1, 1])
                    signal = close.rolling(p).mean() / close - 1.0
                    signal = signal.clip(-1, 1).fillna(0.0)
                    signal.name = self._NAME
                    return signal
        ''')
    else:
        code = textwrap.dedent(f'''\
            """
            {name} — Phi-nance data vendor plugin
            Author  : {author}
            Version : {version}
            """

            from __future__ import annotations
            import pandas as pd
            from phinance.plugins.marketplace import get_marketplace, PluginManifest

            _mp = get_marketplace()

            @_mp.publish(PluginManifest(
                name="{name}",
                version="{version}",
                author="{author}",
                description="{desc}",
                plugin_type="vendor",
                tags=[],
            ))
            class {cls_name}:
                """Fetch OHLCV data from {name}."""

                def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
                    # TODO: implement data fetching
                    raise NotImplementedError("Implement fetch() for {name}")

                def get_name(self) -> str:
                    return "{name}"
        ''')

    return code

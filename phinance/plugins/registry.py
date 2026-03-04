"""
phinance.plugins.registry
===========================

Central plugin registry for third-party indicators and data vendors.

The plugin system allows external packages to register custom:
  • **Indicators** — any class implementing ``BaseIndicator`` protocol
  • **Data vendors** — any callable that returns an OHLCV ``pd.DataFrame``

Registration methods
--------------------
  1. **Direct** — call ``register_indicator()`` / ``register_vendor()``
     from Python code.
  2. **Entry-points** — declare ``phinance.indicators`` or
     ``phinance.vendors`` entry-points in ``pyproject.toml``; call
     ``load_entry_point_plugins()`` to auto-discover them.
  3. **Directory scan** — call ``load_plugin_directory(path)`` to
     import all ``*.py`` files and auto-register decorated classes.

Decorator API
-------------
  @register_indicator_plugin("My Indicator")
  class MyIndicator(BaseIndicator):
      ...

  @register_vendor_plugin("my_vendor")
  def fetch_my_data(symbol, start, end, **kwargs) -> pd.DataFrame:
      ...

Public API
----------
  PluginRegistry       — singleton registry (use ``get_registry()``)
  register_indicator   — register an indicator class
  register_vendor      — register a data-vendor callable
  list_plugins         — return {indicators: [...], vendors: [...]}
  get_indicator_plugin — retrieve a registered indicator by name
  get_vendor_plugin    — retrieve a registered vendor by name
  load_entry_point_plugins — auto-discover via importlib.metadata
  load_plugin_directory    — scan a directory and import plugins
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Type

from phinance.strategies.base import BaseIndicator
from phinance.utils.logging import get_logger

logger = get_logger(__name__)


# ── PluginRegistry ─────────────────────────────────────────────────────────────


class PluginRegistry:
    """
    Central registry for third-party indicators and data vendors.

    Thread-safety note
    ------------------
    The registry is not thread-safe by design (same as the built-in
    indicator catalog). For multi-threaded environments, register all
    plugins at startup before spawning threads.
    """

    def __init__(self) -> None:
        self._indicators: Dict[str, BaseIndicator]  = {}
        self._vendors:    Dict[str, Callable]        = {}
        self._metadata:   Dict[str, Dict[str, Any]] = {}   # name → {version, author, …}

    # ── Indicators ────────────────────────────────────────────────────────────

    def register_indicator(
        self,
        name:      str,
        indicator: "BaseIndicator | Type[BaseIndicator]",
        metadata:  Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a custom indicator.

        Parameters
        ----------
        name      : str                    — unique display name
        indicator : BaseIndicator instance or class
        metadata  : dict, optional         — {version, author, description, …}
        overwrite : bool                   — allow replacing existing registration

        Raises
        ------
        ValueError — if name already registered and overwrite=False
        TypeError  — if indicator does not satisfy BaseIndicator protocol
        """
        if inspect.isclass(indicator):
            indicator = indicator()

        if not isinstance(indicator, BaseIndicator):
            raise TypeError(
                f"Plugin '{name}' must be a BaseIndicator instance, "
                f"got {type(indicator).__name__}"
            )
        if name in self._indicators and not overwrite:
            raise ValueError(
                f"Indicator plugin '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )

        self._indicators[name] = indicator
        self._metadata[f"indicator::{name}"] = metadata or {}
        logger.info("Registered indicator plugin: '%s'", name)

    def get_indicator(self, name: str) -> Optional[BaseIndicator]:
        """Retrieve a registered indicator by name."""
        return self._indicators.get(name)

    def list_indicators(self) -> List[str]:
        """Return all registered indicator plugin names."""
        return sorted(self._indicators.keys())

    # ── Vendors ───────────────────────────────────────────────────────────────

    def register_vendor(
        self,
        name:      str,
        vendor_fn: Callable,
        metadata:  Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a data-vendor callable.

        The callable must have signature:
            ``vendor_fn(symbol: str, start: str, end: str, **kwargs) → pd.DataFrame``

        Parameters
        ----------
        name      : str      — unique vendor name (e.g. ``"polygon"``)
        vendor_fn : callable — data-fetching function
        metadata  : dict, optional
        overwrite : bool
        """
        if not callable(vendor_fn):
            raise TypeError(f"Vendor plugin '{name}' must be callable.")
        if name in self._vendors and not overwrite:
            raise ValueError(
                f"Vendor plugin '{name}' is already registered. "
                "Pass overwrite=True to replace it."
            )
        self._vendors[name] = vendor_fn
        self._metadata[f"vendor::{name}"] = metadata or {}
        logger.info("Registered vendor plugin: '%s'", name)

    def get_vendor(self, name: str) -> Optional[Callable]:
        """Retrieve a registered vendor by name."""
        return self._vendors.get(name)

    def list_vendors(self) -> List[str]:
        """Return all registered vendor plugin names."""
        return sorted(self._vendors.keys())

    # ── Combined ──────────────────────────────────────────────────────────────

    def list_plugins(self) -> Dict[str, List[str]]:
        """Return ``{"indicators": [...], "vendors": [...]}``."""
        return {
            "indicators": self.list_indicators(),
            "vendors":    self.list_vendors(),
        }

    def get_metadata(self, plugin_type: str, name: str) -> Dict[str, Any]:
        """Return metadata for a plugin. plugin_type: 'indicator' or 'vendor'."""
        key = f"{plugin_type}::{name}"
        return dict(self._metadata.get(key, {}))

    def __len__(self) -> int:
        return len(self._indicators) + len(self._vendors)

    def __repr__(self) -> str:
        return (
            f"PluginRegistry("
            f"indicators={len(self._indicators)}, "
            f"vendors={len(self._vendors)})"
        )


# ── Global singleton ──────────────────────────────────────────────────────────

_GLOBAL_REGISTRY: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Return the global singleton PluginRegistry (create if needed)."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = PluginRegistry()
    return _GLOBAL_REGISTRY


def reset_registry() -> None:
    """Reset the global registry (useful in tests)."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = PluginRegistry()


# ── Module-level convenience functions ───────────────────────────────────────


def register_indicator(
    name:      str,
    indicator: "BaseIndicator | Type[BaseIndicator]",
    metadata:  Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """Register an indicator in the global registry."""
    get_registry().register_indicator(name, indicator, metadata=metadata, overwrite=overwrite)


def register_vendor(
    name:      str,
    vendor_fn: Callable,
    metadata:  Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> None:
    """Register a vendor in the global registry."""
    get_registry().register_vendor(name, vendor_fn, metadata=metadata, overwrite=overwrite)


def list_plugins() -> Dict[str, List[str]]:
    """Return all registered plugins from the global registry."""
    return get_registry().list_plugins()


def get_indicator_plugin(name: str) -> Optional[BaseIndicator]:
    """Retrieve a registered indicator plugin from the global registry."""
    return get_registry().get_indicator(name)


def get_vendor_plugin(name: str) -> Optional[Callable]:
    """Retrieve a registered vendor plugin from the global registry."""
    return get_registry().get_vendor(name)


# ── Decorators ────────────────────────────────────────────────────────────────


def register_indicator_plugin(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Callable:
    """
    Class decorator for registering a custom indicator.

    Usage
    -----
        @register_indicator_plugin("My RSI Variant")
        class MyRSI(BaseIndicator):
            def compute(self, df, **params):
                ...
    """
    def decorator(cls: Type[BaseIndicator]) -> Type[BaseIndicator]:
        get_registry().register_indicator(name, cls, metadata=metadata, overwrite=overwrite)
        return cls
    return decorator


def register_vendor_plugin(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Callable:
    """
    Function decorator for registering a custom data vendor.

    Usage
    -----
        @register_vendor_plugin("my_broker")
        def fetch_data(symbol, start, end, **kwargs) -> pd.DataFrame:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        get_registry().register_vendor(name, fn, metadata=metadata, overwrite=overwrite)
        return fn
    return decorator


# ── Entry-point discovery ─────────────────────────────────────────────────────


def load_entry_point_plugins(
    indicator_group: str = "phinance.indicators",
    vendor_group:    str = "phinance.vendors",
) -> Dict[str, int]:
    """
    Auto-discover plugins via Python entry-points.

    Third-party packages declare in ``pyproject.toml``:
        [project.entry-points."phinance.indicators"]
        my_rsi = "my_package.indicators:MyRSI"

    Parameters
    ----------
    indicator_group : str — entry-point group name for indicators
    vendor_group    : str — entry-point group name for vendors

    Returns
    -------
    dict — {"indicators_loaded": N, "vendors_loaded": M}
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        logger.warning("importlib.metadata not available; skipping entry-point discovery")
        return {"indicators_loaded": 0, "vendors_loaded": 0}

    ind_count = 0
    for ep in entry_points(group=indicator_group):
        try:
            cls = ep.load()
            get_registry().register_indicator(ep.name, cls, overwrite=True)
            ind_count += 1
        except Exception as exc:
            logger.warning("Failed to load indicator entry-point '%s': %s", ep.name, exc)

    ven_count = 0
    for ep in entry_points(group=vendor_group):
        try:
            fn = ep.load()
            get_registry().register_vendor(ep.name, fn, overwrite=True)
            ven_count += 1
        except Exception as exc:
            logger.warning("Failed to load vendor entry-point '%s': %s", ep.name, exc)

    logger.info(
        "Entry-point plugins loaded: %d indicators, %d vendors",
        ind_count, ven_count,
    )
    return {"indicators_loaded": ind_count, "vendors_loaded": ven_count}


def load_plugin_directory(
    path: str,
    recursive: bool = False,
) -> Dict[str, int]:
    """
    Import all ``*.py`` files in ``path`` to trigger decorator-based registration.

    Parameters
    ----------
    path      : str  — directory containing plugin files
    recursive : bool — if True, also scan sub-directories

    Returns
    -------
    dict — {"files_loaded": N, "errors": M}
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Plugin directory not found: {path}")

    loaded = 0
    errors = 0

    pattern_dirs = [path]
    if recursive:
        for root, dirs, _ in os.walk(path):
            pattern_dirs.extend(os.path.join(root, d) for d in dirs)

    for dir_path in pattern_dirs:
        for filename in os.listdir(dir_path):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            file_path = os.path.join(dir_path, filename)
            module_name = f"_plugin_{os.path.splitext(filename)[0]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                mod  = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                loaded += 1
                logger.debug("Loaded plugin file: %s", file_path)
            except Exception as exc:
                errors += 1
                logger.warning("Failed to load plugin file '%s': %s", file_path, exc)

    logger.info("Plugin directory scan: %d loaded, %d errors", loaded, errors)
    return {"files_loaded": loaded, "errors": errors}

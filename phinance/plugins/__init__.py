"""
phinance.plugins
=================

Public interface for the Phi-nance plugin system.
"""

from phinance.plugins.registry import (
    PluginRegistry,
    get_registry,
    reset_registry,
    register_indicator,
    register_vendor,
    list_plugins,
    get_indicator_plugin,
    get_vendor_plugin,
    register_indicator_plugin,
    register_vendor_plugin,
    load_entry_point_plugins,
    load_plugin_directory,
)
from phinance.plugins.marketplace import (
    MarketplaceRegistry,
    PluginManifest,
    PluginEntry,
    get_marketplace,
    reset_marketplace,
    scaffold_plugin,
)

__all__ = [
    "PluginRegistry",
    "get_registry",
    "reset_registry",
    "register_indicator",
    "register_vendor",
    "list_plugins",
    "get_indicator_plugin",
    "get_vendor_plugin",
    "register_indicator_plugin",
    "register_vendor_plugin",
    "load_entry_point_plugins",
    "load_plugin_directory",
    # marketplace
    "MarketplaceRegistry",
    "PluginManifest",
    "PluginEntry",
    "get_marketplace",
    "reset_marketplace",
    "scaffold_plugin",
]

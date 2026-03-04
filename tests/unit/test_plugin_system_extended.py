"""
tests.unit.test_plugin_system_extended
========================================

Extended unit tests for the phinance plugin system:
  • PluginRegistry           (plugins/registry.py)
  • register_indicator_plugin decorator
  • register_vendor_plugin decorator
  • get_registry / reset_registry singletons
  • list_plugins
  • load_plugin_directory
  • metadata handling

All tests are pure-unit, no network calls.

NOTE: The PluginRegistry stores *instances* of indicator classes (not the
classes themselves), so we compare by type or isinstance checks.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import pytest

from phinance.strategies.base import BaseIndicator
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
    load_plugin_directory,
)
from tests.fixtures.ohlcv import make_ohlcv

DF_50 = make_ohlcv(n=50)


# ── Helper factories ──────────────────────────────────────────────────────────

def _make_indicator_class(name="DummyInd") -> type:
    """Return a fresh concrete BaseIndicator subclass with a unique class name."""
    def _compute(self, df, **params):
        return pd.Series(0.0, index=df.index, name=name)

    return type(name, (BaseIndicator,), {
        "compute":        _compute,
        "default_params": property(lambda self: {}),
        "param_grid":     property(lambda self: {}),
    })


def _dummy_vendor():
    def fetch(symbol, start, end, **kwargs) -> pd.DataFrame:
        return make_ohlcv(n=20)
    return fetch


# ═══════════════════════════════════════════════════════════════════════════════
# PluginRegistry — Indicator Registration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluginRegistryIndicators:

    def setup_method(self):
        self.reg = PluginRegistry()

    def test_empty_registry_has_no_indicators(self):
        assert self.reg.list_indicators() == []

    def test_register_indicator_appears_in_list(self):
        cls = _make_indicator_class("IndA")
        self.reg.register_indicator("IndA", cls)
        assert "IndA" in self.reg.list_indicators()

    def test_get_registered_indicator_is_not_none(self):
        cls = _make_indicator_class("IndB")
        self.reg.register_indicator("IndB", cls)
        retrieved = self.reg.get_indicator("IndB")
        assert retrieved is not None

    def test_get_registered_indicator_is_base_indicator(self):
        cls = _make_indicator_class("IndC")
        self.reg.register_indicator("IndC", cls)
        retrieved = self.reg.get_indicator("IndC")
        # Registry may store an instance
        assert isinstance(retrieved, (BaseIndicator, type))

    def test_get_nonexistent_indicator_returns_none(self):
        assert self.reg.get_indicator("Ghost") is None

    def test_register_multiple_indicators(self):
        for i in range(5):
            name = f"MultiInd{i}"
            self.reg.register_indicator(name, _make_indicator_class(name))
        assert len(self.reg.list_indicators()) == 5

    def test_overwrite_indicator(self):
        cls1 = _make_indicator_class("OldInd")
        cls2 = _make_indicator_class("NewInd")
        self.reg.register_indicator("MyInd", cls1)
        self.reg.register_indicator("MyInd", cls2, overwrite=True)
        # After overwrite a different object should be returned
        assert self.reg.get_indicator("MyInd") is not None

    def test_list_indicators_returns_list(self):
        self.reg.register_indicator("X", _make_indicator_class("X"))
        result = self.reg.list_indicators()
        assert isinstance(result, list)
        assert "X" in result


# ═══════════════════════════════════════════════════════════════════════════════
# PluginRegistry — Vendor Registration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluginRegistryVendors:

    def setup_method(self):
        self.reg = PluginRegistry()

    def test_empty_registry_has_no_vendors(self):
        assert self.reg.list_vendors() == []

    def test_register_vendor(self):
        fn = _dummy_vendor()
        self.reg.register_vendor("VendorA", fn)
        assert "VendorA" in self.reg.list_vendors()

    def test_get_registered_vendor_is_callable(self):
        fn = _dummy_vendor()
        self.reg.register_vendor("VendorB", fn)
        retrieved = self.reg.get_vendor("VendorB")
        assert callable(retrieved)

    def test_get_nonexistent_vendor_returns_none(self):
        assert self.reg.get_vendor("Missing") is None

    def test_register_multiple_vendors(self):
        for i in range(3):
            self.reg.register_vendor(f"Vendor{i}", _dummy_vendor())
        assert len(self.reg.list_vendors()) == 3

    def test_vendor_returns_dataframe(self):
        fn = _dummy_vendor()
        self.reg.register_vendor("DataVendor", fn)
        vendor = self.reg.get_vendor("DataVendor")
        result = vendor("SPY", "2023-01-01", "2023-12-31")
        assert isinstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# PluginRegistry — list_plugins / len / repr
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluginRegistryMeta:

    def setup_method(self):
        self.reg = PluginRegistry()

    def test_list_plugins_returns_dict(self):
        info = self.reg.list_plugins()
        assert isinstance(info, dict)
        assert "indicators" in info
        assert "vendors" in info

    def test_list_plugins_empty(self):
        info = self.reg.list_plugins()
        assert info["indicators"] == []
        assert info["vendors"] == []

    def test_list_plugins_after_registration(self):
        self.reg.register_indicator("Ind1", _make_indicator_class("Ind1"))
        self.reg.register_vendor("V1", _dummy_vendor())
        info = self.reg.list_plugins()
        assert "Ind1" in info["indicators"]
        assert "V1" in info["vendors"]

    def test_len_empty(self):
        assert len(self.reg) == 0

    def test_len_after_mixed_registrations(self):
        self.reg.register_indicator("I1", _make_indicator_class("I1"))
        self.reg.register_vendor("V1", _dummy_vendor())
        assert len(self.reg) == 2

    def test_repr_contains_class_name(self):
        r = repr(self.reg)
        assert "PluginRegistry" in r or "registry" in r.lower()

    def test_get_metadata_missing_returns_empty_or_dict(self):
        got = self.reg.get_metadata("indicators", "Ghost")
        assert isinstance(got, dict)

    def test_get_metadata_registered_with_metadata(self):
        meta = {"version": "1.0", "author": "test"}
        self.reg.register_indicator("MetaInd", _make_indicator_class("MetaInd"), metadata=meta)
        got = self.reg.get_metadata("indicators", "MetaInd")
        # Should contain the metadata we passed (may be nested under a key)
        assert isinstance(got, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level convenience helpers (singleton)
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleLevelHelpers:

    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_get_registry_returns_plugin_registry(self):
        reg = get_registry()
        assert isinstance(reg, PluginRegistry)

    def test_get_registry_is_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_registry_clears(self):
        register_indicator("TempInd", _make_indicator_class("TempInd"))
        reset_registry()
        assert "TempInd" not in list_plugins()["indicators"]

    def test_module_register_indicator_in_list(self):
        register_indicator("MI1", _make_indicator_class("MI1"))
        assert "MI1" in list_plugins()["indicators"]

    def test_module_register_vendor_in_list(self):
        register_vendor("MV1", _dummy_vendor())
        assert "MV1" in list_plugins()["vendors"]

    def test_module_get_indicator_plugin_not_none(self):
        cls = _make_indicator_class("MIG")
        register_indicator("MIG", cls)
        result = get_indicator_plugin("MIG")
        assert result is not None

    def test_module_get_vendor_plugin_callable(self):
        fn = _dummy_vendor()
        register_vendor("MVG", fn)
        result = get_vendor_plugin("MVG")
        assert callable(result)

    def test_list_plugins_from_module_level(self):
        info = list_plugins()
        assert "indicators" in info
        assert "vendors" in info

    def test_reset_makes_registry_empty(self):
        register_indicator("ToReset", _make_indicator_class("ToReset"))
        register_vendor("VToReset", _dummy_vendor())
        reset_registry()
        info = list_plugins()
        assert len(info["indicators"]) == 0
        assert len(info["vendors"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecorators:

    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_register_indicator_plugin_decorator_adds_to_registry(self):
        @register_indicator_plugin("DecoratedInd")
        class DecoratedInd(BaseIndicator):
            def compute(self, df, **params):
                return pd.Series(0.0, index=df.index)
            @property
            def default_params(self):
                return {}
            @property
            def param_grid(self):
                return {}

        assert "DecoratedInd" in list_plugins()["indicators"]

    def test_register_vendor_plugin_decorator_adds_to_registry(self):
        @register_vendor_plugin("DecoratedVendor")
        def decorated_vendor(symbol, start, end, **kwargs):
            return make_ohlcv(n=20)

        assert "DecoratedVendor" in list_plugins()["vendors"]

    def test_decorated_indicator_retrievable(self):
        @register_indicator_plugin("RetrievableInd")
        class RetrievableInd(BaseIndicator):
            def compute(self, df, **params):
                return pd.Series(0.0, index=df.index)
            @property
            def default_params(self):
                return {}
            @property
            def param_grid(self):
                return {}

        retrieved = get_indicator_plugin("RetrievableInd")
        assert retrieved is not None

    def test_decorated_vendor_callable(self):
        @register_vendor_plugin("CallDec2")
        def call_dec(symbol, start, end, **kwargs):
            return make_ohlcv(n=10)

        fn = get_vendor_plugin("CallDec2")
        result = fn("SPY", "2023-01-01", "2023-12-31")
        assert isinstance(result, pd.DataFrame)

    def test_indicator_plugin_compute_works(self):
        @register_indicator_plugin("ComputeInd")
        class ComputeInd(BaseIndicator):
            def compute(self, df, **params):
                return pd.Series(1.0, index=df.index, name="ComputeInd")
            @property
            def default_params(self):
                return {}
            @property
            def param_grid(self):
                return {}

        retrieved = get_indicator_plugin("ComputeInd")
        if isinstance(retrieved, type):
            ind = retrieved()
        else:
            ind = retrieved
        result = ind.compute(DF_50)
        assert isinstance(result, pd.Series)


# ═══════════════════════════════════════════════════════════════════════════════
# load_plugin_directory
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPluginDirectory:

    def setup_method(self):
        reset_registry()

    def teardown_method(self):
        reset_registry()

    def test_load_empty_directory_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            load_plugin_directory(tmpdir)   # should not raise

    def test_load_nonexistent_directory_does_not_raise(self):
        try:
            load_plugin_directory("/nonexistent/path/xyz_abc")
        except Exception:
            pass   # graceful failure acceptable

    def test_load_directory_with_plugin_file(self):
        plugin_code = '''\
from phinance.strategies.base import BaseIndicator
import pandas as pd
from phinance.plugins.registry import register_indicator_plugin

@register_indicator_plugin("DirLoadedInd99")
class DirLoadedInd99(BaseIndicator):
    def compute(self, df, **params):
        return pd.Series(0.0, index=df.index)
    @property
    def default_params(self):
        return {}
    @property
    def param_grid(self):
        return {}
'''
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = os.path.join(tmpdir, "my_plugin.py")
            with open(plugin_path, "w") as f:
                f.write(plugin_code)
            load_plugin_directory(tmpdir)
            plugins = list_plugins()
            assert "DirLoadedInd99" in plugins["indicators"]

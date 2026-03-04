"""
tests.unit.test_plugin_system
================================

Comprehensive tests for Phase 9.2 — Plugin System:
  • PluginRegistry (register, retrieve, list, overwrite, metadata)
  • Global singleton (get_registry, reset_registry)
  • Module-level convenience functions
  • Decorator-based registration (@register_indicator_plugin, @register_vendor_plugin)
  • load_entry_point_plugins (mocked)
  • load_plugin_directory
  • Error handling (duplicate names, wrong types)
"""

from __future__ import annotations

import os
import pytest
import pandas as pd
import numpy as np

from tests.fixtures.ohlcv import make_ohlcv

DF = make_ohlcv(n=100)


# ── Minimal indicator for testing ─────────────────────────────────────────────

def _make_indicator(name: str = "TestPlugin"):
    from phinance.strategies.base import BaseIndicator

    class _DummyIndicator(BaseIndicator):
        def compute(self, df: pd.DataFrame, **params) -> pd.Series:
            return pd.Series(0.5, index=df.index, name=name)

    _DummyIndicator.name = name
    return _DummyIndicator()


def _make_vendor():
    def _vendor(symbol: str, start: str, end: str, **kwargs) -> pd.DataFrame:
        return make_ohlcv(n=50)
    return _vendor


# ═══════════════════════════════════════════════════════════════════════════════
# PluginRegistry — indicators
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluginRegistryIndicators:

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        """Use a fresh registry for each test (not the global one)."""
        from phinance.plugins.registry import PluginRegistry
        self.reg = PluginRegistry()

    def test_register_and_list(self):
        ind = _make_indicator("Alpha")
        self.reg.register_indicator("Alpha", ind)
        assert "Alpha" in self.reg.list_indicators()

    def test_register_returns_none(self):
        result = self.reg.register_indicator("Beta", _make_indicator("Beta"))
        assert result is None

    def test_get_registered_indicator(self):
        ind = _make_indicator("Gamma")
        self.reg.register_indicator("Gamma", ind)
        fetched = self.reg.get_indicator("Gamma")
        assert fetched is not None
        assert fetched.name == "Gamma"

    def test_get_unknown_returns_none(self):
        assert self.reg.get_indicator("unknown") is None

    def test_list_indicators_sorted(self):
        self.reg.register_indicator("Zeta", _make_indicator("Zeta"))
        self.reg.register_indicator("Alpha", _make_indicator("Alpha"))
        names = self.reg.list_indicators()
        assert names == sorted(names)

    def test_duplicate_raises_without_overwrite(self):
        self.reg.register_indicator("Dup", _make_indicator("Dup"))
        with pytest.raises(ValueError, match="already registered"):
            self.reg.register_indicator("Dup", _make_indicator("Dup"))

    def test_overwrite_allowed(self):
        ind1 = _make_indicator("Overwrite")
        ind2 = _make_indicator("Overwrite")
        self.reg.register_indicator("Overwrite", ind1)
        self.reg.register_indicator("Overwrite", ind2, overwrite=True)
        assert self.reg.get_indicator("Overwrite") is ind2

    def test_register_class_auto_instantiates(self):
        from phinance.strategies.base import BaseIndicator

        class _ClassPlugin(BaseIndicator):
            name = "ClassPlugin"
            def compute(self, df, **params):
                return pd.Series(0.0, index=df.index)

        self.reg.register_indicator("ClassPlugin", _ClassPlugin)
        fetched = self.reg.get_indicator("ClassPlugin")
        assert isinstance(fetched, BaseIndicator)

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self.reg.register_indicator("Bad", "not_an_indicator")

    def test_indicator_compute_works(self):
        ind = _make_indicator("Compute")
        self.reg.register_indicator("Compute", ind)
        result = self.reg.get_indicator("Compute").compute(DF)
        assert isinstance(result, pd.Series)
        assert len(result) == len(DF)

    def test_metadata_stored_and_retrieved(self):
        meta = {"version": "1.0", "author": "TestBot"}
        self.reg.register_indicator("MetaInd", _make_indicator("MetaInd"), metadata=meta)
        retrieved = self.reg.get_metadata("indicator", "MetaInd")
        assert retrieved["version"] == "1.0"
        assert retrieved["author"]  == "TestBot"

    def test_metadata_empty_if_not_provided(self):
        self.reg.register_indicator("NoMeta", _make_indicator("NoMeta"))
        assert self.reg.get_metadata("indicator", "NoMeta") == {}


# ═══════════════════════════════════════════════════════════════════════════════
# PluginRegistry — vendors
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluginRegistryVendors:

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        from phinance.plugins.registry import PluginRegistry
        self.reg = PluginRegistry()

    def test_register_and_list(self):
        self.reg.register_vendor("vendor_a", _make_vendor())
        assert "vendor_a" in self.reg.list_vendors()

    def test_get_registered_vendor(self):
        fn = _make_vendor()
        self.reg.register_vendor("vendor_b", fn)
        fetched = self.reg.get_vendor("vendor_b")
        assert fetched is fn

    def test_get_unknown_vendor_returns_none(self):
        assert self.reg.get_vendor("no_such_vendor") is None

    def test_vendor_callable_works(self):
        self.reg.register_vendor("vendor_c", _make_vendor())
        fn = self.reg.get_vendor("vendor_c")
        df = fn("SPY", "2023-01-01", "2023-06-01")
        assert isinstance(df, pd.DataFrame)
        assert "close" in df.columns

    def test_duplicate_raises_without_overwrite(self):
        self.reg.register_vendor("dup_vendor", _make_vendor())
        with pytest.raises(ValueError, match="already registered"):
            self.reg.register_vendor("dup_vendor", _make_vendor())

    def test_overwrite_vendor(self):
        fn1 = _make_vendor()
        fn2 = _make_vendor()
        self.reg.register_vendor("ov_vendor", fn1)
        self.reg.register_vendor("ov_vendor", fn2, overwrite=True)
        assert self.reg.get_vendor("ov_vendor") is fn2

    def test_non_callable_raises(self):
        with pytest.raises(TypeError):
            self.reg.register_vendor("bad_vendor", "not_callable")

    def test_vendor_metadata(self):
        meta = {"description": "Test vendor", "auth_required": False}
        self.reg.register_vendor("meta_vendor", _make_vendor(), metadata=meta)
        retrieved = self.reg.get_metadata("vendor", "meta_vendor")
        assert retrieved["auth_required"] is False

    def test_list_vendors_sorted(self):
        self.reg.register_vendor("z_vendor", _make_vendor())
        self.reg.register_vendor("a_vendor", _make_vendor())
        names = self.reg.list_vendors()
        assert names == sorted(names)


# ═══════════════════════════════════════════════════════════════════════════════
# PluginRegistry — combined
# ═══════════════════════════════════════════════════════════════════════════════

class TestPluginRegistryCombined:

    @pytest.fixture(autouse=True)
    def fresh_registry(self):
        from phinance.plugins.registry import PluginRegistry
        self.reg = PluginRegistry()

    def test_list_plugins_returns_both(self):
        self.reg.register_indicator("I1", _make_indicator("I1"))
        self.reg.register_vendor("V1", _make_vendor())
        plugins = self.reg.list_plugins()
        assert "indicators" in plugins
        assert "vendors"    in plugins
        assert "I1" in plugins["indicators"]
        assert "V1" in plugins["vendors"]

    def test_len_counts_both(self):
        self.reg.register_indicator("X", _make_indicator("X"))
        self.reg.register_vendor("Y", _make_vendor())
        assert len(self.reg) == 2

    def test_repr_contains_counts(self):
        r = repr(self.reg)
        assert "PluginRegistry" in r
        assert "indicators" in r
        assert "vendors"    in r


# ═══════════════════════════════════════════════════════════════════════════════
# Global singleton
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalRegistry:

    @pytest.fixture(autouse=True)
    def reset(self):
        from phinance.plugins.registry import reset_registry
        reset_registry()
        yield
        reset_registry()

    def test_get_registry_returns_same_instance(self):
        from phinance.plugins.registry import get_registry
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_reset_creates_new_instance(self):
        from phinance.plugins.registry import get_registry, reset_registry
        r1 = get_registry()
        reset_registry()
        r2 = get_registry()
        assert r1 is not r2

    def test_module_level_register_indicator(self):
        from phinance.plugins.registry import register_indicator, list_plugins, reset_registry
        reset_registry()
        register_indicator("GlobalInd", _make_indicator("GlobalInd"))
        plugins = list_plugins()
        assert "GlobalInd" in plugins["indicators"]

    def test_module_level_register_vendor(self):
        from phinance.plugins.registry import register_vendor, list_plugins, reset_registry
        reset_registry()
        register_vendor("GlobalVen", _make_vendor())
        plugins = list_plugins()
        assert "GlobalVen" in plugins["vendors"]

    def test_get_indicator_plugin(self):
        from phinance.plugins.registry import register_indicator, get_indicator_plugin, reset_registry
        reset_registry()
        ind = _make_indicator("GetTest")
        register_indicator("GetTest", ind)
        fetched = get_indicator_plugin("GetTest")
        assert fetched is ind

    def test_get_vendor_plugin(self):
        from phinance.plugins.registry import register_vendor, get_vendor_plugin, reset_registry
        reset_registry()
        fn = _make_vendor()
        register_vendor("GetVendorTest", fn)
        assert get_vendor_plugin("GetVendorTest") is fn


# ═══════════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecorators:

    @pytest.fixture(autouse=True)
    def reset(self):
        from phinance.plugins.registry import reset_registry
        reset_registry()
        yield
        reset_registry()

    def test_register_indicator_plugin_decorator(self):
        from phinance.plugins.registry import register_indicator_plugin, get_indicator_plugin
        from phinance.strategies.base import BaseIndicator

        @register_indicator_plugin("DecoratedInd")
        class _Dec(BaseIndicator):
            name = "DecoratedInd"
            def compute(self, df, **p):
                return pd.Series(1.0, index=df.index)

        fetched = get_indicator_plugin("DecoratedInd")
        assert fetched is not None
        assert isinstance(fetched, BaseIndicator)

    def test_register_vendor_plugin_decorator(self):
        from phinance.plugins.registry import register_vendor_plugin, get_vendor_plugin

        @register_vendor_plugin("DecoratedVendor")
        def _fetch(symbol, start, end, **kw):
            return make_ohlcv(n=20)

        fetched = get_vendor_plugin("DecoratedVendor")
        assert fetched is not None
        assert callable(fetched)

    def test_decorator_preserves_class(self):
        from phinance.plugins.registry import register_indicator_plugin
        from phinance.strategies.base import BaseIndicator

        @register_indicator_plugin("Preserved")
        class _Pres(BaseIndicator):
            name = "Preserved"
            def compute(self, df, **p):
                return pd.Series(0.0, index=df.index)

        assert _Pres is not None
        assert issubclass(_Pres, BaseIndicator)

    def test_decorator_preserves_function(self):
        from phinance.plugins.registry import register_vendor_plugin

        @register_vendor_plugin("PreservedVendor")
        def _pv(symbol, start, end, **kw):
            return make_ohlcv(n=10)

        assert callable(_pv)


# ═══════════════════════════════════════════════════════════════════════════════
# load_plugin_directory
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadPluginDirectory:

    @pytest.fixture(autouse=True)
    def reset(self):
        from phinance.plugins.registry import reset_registry
        reset_registry()
        yield
        reset_registry()

    def test_nonexistent_dir_raises(self):
        from phinance.plugins.registry import load_plugin_directory
        with pytest.raises(NotADirectoryError):
            load_plugin_directory("/nonexistent/path/xyz")

    def test_empty_dir_loads_zero(self, tmp_path):
        from phinance.plugins.registry import load_plugin_directory
        result = load_plugin_directory(str(tmp_path))
        assert result["files_loaded"] == 0
        assert result["errors"] == 0

    def test_loads_valid_plugin_file(self, tmp_path):
        from phinance.plugins.registry import load_plugin_directory, get_indicator_plugin, reset_registry
        reset_registry()
        # Create a plugin file that registers an indicator
        plugin_code = '''
from phinance.plugins.registry import register_indicator_plugin
from phinance.strategies.base import BaseIndicator
import pandas as pd

@register_indicator_plugin("FilePluginInd", overwrite=True)
class _FilePlugin(BaseIndicator):
    name = "FilePluginInd"
    def compute(self, df, **p):
        return pd.Series(0.0, index=df.index)
'''
        (tmp_path / "my_plugin.py").write_text(plugin_code)
        result = load_plugin_directory(str(tmp_path))
        assert result["files_loaded"] == 1
        assert result["errors"] == 0

    def test_skips_dunder_files(self, tmp_path):
        from phinance.plugins.registry import load_plugin_directory
        (tmp_path / "__init__.py").write_text("# init")
        result = load_plugin_directory(str(tmp_path))
        assert result["files_loaded"] == 0

    def test_error_file_counted(self, tmp_path):
        from phinance.plugins.registry import load_plugin_directory
        (tmp_path / "bad_plugin.py").write_text("raise RuntimeError('bad')")
        result = load_plugin_directory(str(tmp_path))
        assert result["errors"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# load_entry_point_plugins (mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadEntryPointPlugins:

    @pytest.fixture(autouse=True)
    def reset(self):
        from phinance.plugins.registry import reset_registry
        reset_registry()
        yield
        reset_registry()

    def test_returns_dict_with_counts(self):
        from phinance.plugins.registry import load_entry_point_plugins
        result = load_entry_point_plugins(
            indicator_group="phinance.nonexistent.indicators",
            vendor_group="phinance.nonexistent.vendors",
        )
        assert "indicators_loaded" in result
        assert "vendors_loaded"    in result

    def test_no_plugins_for_unknown_group(self):
        from phinance.plugins.registry import load_entry_point_plugins
        result = load_entry_point_plugins(
            indicator_group="phinance.nonexistent.indicators",
            vendor_group="phinance.nonexistent.vendors",
        )
        assert result["indicators_loaded"] == 0
        assert result["vendors_loaded"]    == 0

"""
tests/unit/test_plugin_marketplace.py
========================================

Comprehensive unit tests for phinance.plugins.marketplace.

Covers:
  - PluginManifest (creation, validation, missing deps, to_dict, repr)
  - PluginEntry (creation, to_dict, repr)
  - MarketplaceRegistry (init, publish decorator, register,
    get, list_all, list_versions, search, catalogue, len, repr)
  - MarketplaceRegistry conflict detection + overwrite
  - MarketplaceRegistry vendor plugin registration
  - get_marketplace / reset_marketplace singleton
  - scaffold_plugin (indicator + vendor types)
"""

from __future__ import annotations

import pytest

from phinance.strategies.base import BaseIndicator
import pandas as pd
import numpy as np

from phinance.plugins.marketplace import (
    PluginManifest,
    PluginEntry,
    MarketplaceRegistry,
    get_marketplace,
    reset_marketplace,
    scaffold_plugin,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_mp() -> MarketplaceRegistry:
    reset_marketplace()
    return MarketplaceRegistry()


def _dummy_indicator(ind_name: str = "DummyInd") -> type:
    """Create a concrete BaseIndicator subclass with the given name (no closure bug)."""

    def _compute(self, df, params=None):
        return df["close"].rolling(5).mean().rename(ind_name)

    _cls = type(
        f"_Ind_{ind_name}",
        (BaseIndicator,),
        {
            "name":       property(lambda self: ind_name),
            "param_grid": property(lambda self: {}),
            "compute":    _compute,
        },
    )
    return _cls


def _dummy_vendor() -> type:
    class _Vendor:
        def get_name(self): return "DummyVendor"
    return _Vendor


# ── PluginManifest ────────────────────────────────────────────────────────────


class TestPluginManifest:
    def test_default_creation(self):
        m = PluginManifest()
        assert m.version == "0.1.0"
        assert m.author == "Anonymous"
        assert m.plugin_type == "indicator"
        assert m.license == "MIT"
        assert isinstance(m.plugin_id, str)

    def test_custom_creation(self):
        m = PluginManifest(name="MyPlugin", version="2.1.0", author="Bob")
        assert m.name == "MyPlugin"
        assert m.version == "2.1.0"
        assert m.author == "Bob"

    def test_valid_version_true(self):
        assert PluginManifest(version="1.2.3").is_valid_version()

    def test_valid_version_false_alpha(self):
        assert not PluginManifest(version="1.2.3a").is_valid_version()

    def test_valid_version_false_two_parts(self):
        assert not PluginManifest(version="1.2").is_valid_version()

    def test_missing_deps_none(self):
        m = PluginManifest(requires=["numpy", "pandas"])
        assert m.missing_dependencies() == []

    def test_missing_deps_nonexistent(self):
        m = PluginManifest(requires=["nonexistent_package_xyz_abc"])
        missing = m.missing_dependencies()
        assert "nonexistent_package_xyz_abc" in missing

    def test_to_dict_keys(self):
        m = PluginManifest(name="P", version="1.0.0")
        d = m.to_dict()
        for k in ("name", "version", "author", "description", "plugin_type",
                  "requires", "tags", "homepage", "license", "plugin_id"):
            assert k in d

    def test_repr(self):
        m = PluginManifest(name="MyPlugin", version="1.0.0")
        r = repr(m)
        assert "PluginManifest" in r
        assert "MyPlugin" in r
        assert "1.0.0" in r

    def test_unique_plugin_ids(self):
        ids = {PluginManifest().plugin_id for _ in range(20)}
        assert len(ids) == 20

    def test_tags_list(self):
        m = PluginManifest(tags=["momentum", "trend"])
        assert m.tags == ["momentum", "trend"]


# ── PluginEntry ───────────────────────────────────────────────────────────────


class TestPluginEntry:
    def test_creation(self):
        m = PluginManifest(name="E", version="1.0.0")
        e = PluginEntry(manifest=m, plugin_cls=object, installed=True)
        assert e.manifest is m
        assert e.installed

    def test_to_dict_keys(self):
        m = PluginManifest(name="E2", version="1.0.0")
        e = PluginEntry(manifest=m, plugin_cls=object)
        d = e.to_dict()
        assert "name" in d
        assert "installed" in d

    def test_repr(self):
        m = PluginManifest(name="E3", version="0.5.0")
        e = PluginEntry(manifest=m, plugin_cls=object, installed=False)
        r = repr(e)
        assert "PluginEntry" in r
        assert "E3" in r


# ── MarketplaceRegistry registration ─────────────────────────────────────────


class TestMarketplaceRegistration:
    def test_register_indicator(self):
        mp = _make_mp()
        cls = _dummy_indicator("Ind1")
        m = PluginManifest(name="Ind1", version="1.0.0", plugin_type="indicator")
        entry = mp.register(cls, m)
        assert isinstance(entry, PluginEntry)

    def test_len_after_register(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("L1"), PluginManifest(name="L1", version="1.0.0"))
        assert len(mp) == 1

    def test_len_multiple(self):
        mp = _make_mp()
        for i in range(5):
            mp.register(_dummy_indicator(f"MI{i}"), PluginManifest(name=f"MI{i}", version="1.0.0"))
        assert len(mp) == 5

    def test_installed_true_for_indicator(self):
        mp = _make_mp()
        cls = _dummy_indicator("Inst1")
        e = mp.register(cls, PluginManifest(name="Inst1", version="1.0.0", plugin_type="indicator"))
        assert e.installed

    def test_empty_name_raises(self):
        mp = _make_mp()
        with pytest.raises(ValueError, match="name must not be empty"):
            mp.register(object, PluginManifest(name="", version="1.0.0"))

    def test_invalid_version_raises(self):
        mp = _make_mp()
        with pytest.raises(ValueError, match="Invalid version"):
            mp.register(object, PluginManifest(name="Bad", version="bad"))

    def test_duplicate_raises(self):
        mp = _make_mp()
        m = PluginManifest(name="Dup", version="1.0.0")
        mp.register(_dummy_indicator("Dup"), m)
        with pytest.raises(ValueError, match="already registered"):
            mp.register(_dummy_indicator("Dup"), PluginManifest(name="Dup", version="1.0.0"))

    def test_overwrite_allowed(self):
        mp = _make_mp()
        cls = _dummy_indicator("Ow1")
        mp.register(cls, PluginManifest(name="Ow1", version="1.0.0"))
        mp.register(cls, PluginManifest(name="Ow1", version="1.0.0"), overwrite=True)
        assert len(mp) == 1

    def test_different_versions_both_registered(self):
        mp = _make_mp()
        cls = _dummy_indicator("MV1")
        mp.register(cls, PluginManifest(name="MV1", version="1.0.0"))
        mp.register(cls, PluginManifest(name="MV1", version="2.0.0"))
        assert mp.list_versions("MV1") == ["1.0.0", "2.0.0"]


# ── publish decorator ─────────────────────────────────────────────────────────


class TestPublishDecorator:
    def test_publish_returns_class(self):
        mp = _make_mp()

        @mp.publish(PluginManifest(name="Dec1", version="1.0.0"))
        class Dec1Indicator(BaseIndicator):
            @property
            def name(self): return "Dec1"
            @property
            def param_grid(self): return {}
            def compute(self, df, params=None): return df["close"]

        assert Dec1Indicator is not None
        assert len(mp) == 1

    def test_published_class_unchanged(self):
        mp = _make_mp()

        @mp.publish(PluginManifest(name="Dec2", version="1.0.0"))
        class Dec2Indicator(BaseIndicator):
            MARKER = "original"
            @property
            def name(self): return "Dec2"
            @property
            def param_grid(self): return {}
            def compute(self, df, params=None): return df["close"]

        assert Dec2Indicator.MARKER == "original"


# ── get / list ────────────────────────────────────────────────────────────────


class TestGet:
    def test_get_by_name(self):
        mp = _make_mp()
        cls = _dummy_indicator("G1")
        mp.register(cls, PluginManifest(name="G1", version="1.0.0"))
        e = mp.get("G1")
        assert e is not None
        assert e.manifest.name == "G1"

    def test_get_missing_returns_none(self):
        mp = _make_mp()
        assert mp.get("Nonexistent") is None

    def test_get_specific_version(self):
        mp = _make_mp()
        cls = _dummy_indicator("GV1")
        mp.register(cls, PluginManifest(name="GV1", version="1.0.0"))
        mp.register(cls, PluginManifest(name="GV1", version="2.0.0"))
        e = mp.get("GV1", version="1.0.0")
        assert e.manifest.version == "1.0.0"

    def test_get_latest_version(self):
        mp = _make_mp()
        cls = _dummy_indicator("GL1")
        mp.register(cls, PluginManifest(name="GL1", version="1.0.0"))
        mp.register(cls, PluginManifest(name="GL1", version="2.0.0"))
        e = mp.get("GL1")
        assert e.manifest.version == "2.0.0"

    def test_list_all_returns_latest(self):
        mp = _make_mp()
        cls = _dummy_indicator("LA1")
        mp.register(cls, PluginManifest(name="LA1", version="1.0.0"))
        mp.register(cls, PluginManifest(name="LA1", version="2.0.0"))
        all_entries = mp.list_all()
        versions = [e.manifest.version for e in all_entries if e.manifest.name == "LA1"]
        assert versions == ["2.0.0"]


# ── list_versions ─────────────────────────────────────────────────────────────


class TestListVersions:
    def test_single_version(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("LV1"), PluginManifest(name="LV1", version="1.0.0"))
        assert mp.list_versions("LV1") == ["1.0.0"]

    def test_multiple_versions_sorted(self):
        mp = _make_mp()
        cls = _dummy_indicator("LV2")
        mp.register(cls, PluginManifest(name="LV2", version="1.0.0"))
        mp.register(cls, PluginManifest(name="LV2", version="1.5.0"))
        mp.register(cls, PluginManifest(name="LV2", version="2.0.0"))
        vs = mp.list_versions("LV2")
        assert vs == ["1.0.0", "1.5.0", "2.0.0"]

    def test_missing_name(self):
        mp = _make_mp()
        assert mp.list_versions("NoSuch") == []


# ── search ────────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_by_name(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("MySuperRSI"),
                    PluginManifest(name="MySuperRSI", version="1.0.0", description="Super RSI"))
        results = mp.search(query="SuperRSI")
        assert any(e.manifest.name == "MySuperRSI" for e in results)

    def test_search_by_description(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("S1"),
                    PluginManifest(name="S1", version="1.0.0", description="momentum oscillator"))
        results = mp.search(query="momentum")
        assert len(results) > 0

    def test_search_by_type(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("ST1"),
                    PluginManifest(name="ST1", version="1.0.0", plugin_type="indicator"))
        mp.register(_dummy_vendor(),
                    PluginManifest(name="SV1", version="1.0.0", plugin_type="vendor"))
        results = mp.search(plugin_type="vendor")
        assert all(e.manifest.plugin_type == "vendor" for e in results)

    def test_search_by_tags(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("Tag1"),
                    PluginManifest(name="Tag1", version="1.0.0", tags=["momentum", "rsi"]))
        mp.register(_dummy_indicator("Tag2"),
                    PluginManifest(name="Tag2", version="1.0.0", tags=["trend"]))
        results = mp.search(tags=["rsi"])
        names = [e.manifest.name for e in results]
        assert "Tag1" in names
        assert "Tag2" not in names

    def test_search_empty_returns_all(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("SA1"), PluginManifest(name="SA1", version="1.0.0"))
        mp.register(_dummy_indicator("SA2"), PluginManifest(name="SA2", version="1.0.0"))
        assert len(mp.search()) == 2

    def test_search_no_match(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("NoMatch"), PluginManifest(name="NoMatch", version="1.0.0"))
        assert mp.search(query="xyznonexistent") == []


# ── catalogue ─────────────────────────────────────────────────────────────────


class TestCatalogue:
    def test_returns_list_of_dicts(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("C1"), PluginManifest(name="C1", version="1.0.0"))
        cat = mp.catalogue()
        assert isinstance(cat, list)
        assert all(isinstance(d, dict) for d in cat)

    def test_catalogue_length(self):
        mp = _make_mp()
        for i in range(3):
            mp.register(_dummy_indicator(f"CL{i}"), PluginManifest(name=f"CL{i}", version="1.0.0"))
        assert len(mp.catalogue()) == 3

    def test_catalogue_keys(self):
        mp = _make_mp()
        mp.register(_dummy_indicator("CK1"), PluginManifest(name="CK1", version="1.0.0", author="Alice"))
        d = mp.catalogue()[0]
        assert d["name"] == "CK1"
        assert d["author"] == "Alice"


# ── Singleton ─────────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_marketplace_returns_same(self):
        reset_marketplace()
        m1 = get_marketplace()
        m2 = get_marketplace()
        assert m1 is m2

    def test_reset_marketplace(self):
        reset_marketplace()
        m1 = get_marketplace()
        reset_marketplace()
        m2 = get_marketplace()
        assert m1 is not m2

    def test_repr(self):
        mp = _make_mp()
        assert "MarketplaceRegistry" in repr(mp)


# ── scaffold_plugin ───────────────────────────────────────────────────────────


class TestScaffoldPlugin:
    def test_indicator_scaffold(self):
        code = scaffold_plugin("My Custom RSI", plugin_type="indicator")
        assert "My Custom RSI" in code
        assert "BaseIndicator" in code
        assert "compute" in code

    def test_vendor_scaffold(self):
        code = scaffold_plugin("My Vendor", plugin_type="vendor")
        assert "My Vendor" in code
        assert "fetch" in code

    def test_version_in_scaffold(self):
        code = scaffold_plugin("TestP", version="2.3.0")
        assert "2.3.0" in code

    def test_author_in_scaffold(self):
        code = scaffold_plugin("TestP2", author="Dr. Test")
        assert "Dr. Test" in code

    def test_returns_string(self):
        code = scaffold_plugin("X")
        assert isinstance(code, str)
        assert len(code) > 50

    def test_scaffold_is_valid_python(self):
        code = scaffold_plugin("SyntaxTest", plugin_type="indicator")
        # Should compile without error
        try:
            compile(code, "<scaffold>", "exec")
            valid = True
        except SyntaxError:
            valid = False
        assert valid

    def test_vendor_scaffold_valid_python(self):
        code = scaffold_plugin("VendorTest", plugin_type="vendor")
        try:
            compile(code, "<scaffold>", "exec")
            valid = True
        except SyntaxError:
            valid = False
        assert valid

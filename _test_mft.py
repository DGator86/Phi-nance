"""
Market Field Theory — Smoke Test
Run with: python _test_mft.py
"""
import sys
import traceback

sys.path.insert(0, r"c:\Users\Darrin Vogeli\OneDrive - Penetron\Desktop\Phi-nance-1")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


failed = []


def check(label, fn):
    try:
        result = fn()
        print(f"  OK  {label}")
        return result
    except Exception as e:
        failed.append((label, str(e)))
        print(f"  FAIL {label}: {e}")
        traceback.print_exc()
        return None


# ── 1. Imports ────────────────────────────────────────────────────────────────
section("1. Imports")

imports = check(
    "Import all from regime_engine",
    lambda: __import__("regime_engine", fromlist=[
        "RegimeEngine", "UniverseScanner", "FeatureEngine",
        "GammaSurface", "PolygonL2Client", "PolygonRestClient",
        "load_config", "simulate_ohlcv",
    ])
)

if imports is None:
    print("\nIMPORT FAILED — aborting tests.")
    sys.exit(1)

from regime_engine import (
    RegimeEngine, UniverseScanner, FeatureEngine,
    GammaSurface, PolygonL2Client, PolygonRestClient,
    load_config, simulate_ohlcv,
)

# ── 2. Config ─────────────────────────────────────────────────────────────────
section("2. Configuration")

cfg = check("load_config()", load_config)
if cfg:
    gam_en = cfg.get("gamma", {}).get("enabled")
    pol_en = cfg.get("polygon", {}).get("enabled")
    dis_w  = cfg.get("features", {}).get("dissipation_window")
    print(f"       gamma.enabled={gam_en}, polygon.enabled={pol_en}, dissipation_window={dis_w}")

# ── 3. Feature Engine — MFT L2-proxy features ─────────────────────────────────
section("3. Phase 1 — L2-Proxy Features")

ohlcv = simulate_ohlcv(n_bars=500, seed=42)
fe = FeatureEngine(cfg["features"])
feats = check("FeatureEngine.compute()", lambda: fe.compute(ohlcv))

if feats is not None:
    MFT_COLS = [
        "d_lambda", "mass", "d_mass_dt",
        "ofi_proxy", "absorption_score", "dissipation_proxy",
    ]
    for col in MFT_COLS:
        if col in feats.columns:
            nans = feats[col].isna().sum()
            non_nan = feats[col].notna().sum()
            print(f"       {col:25s}: {non_nan:4d} values, {nans:3d} NaNs")
        else:
            failed.append((f"Feature {col}", "column missing"))
            print(f"  FAIL Feature {col}: MISSING")

# ── 4. Full RegimeEngine ───────────────────────────────────────────────────────
section("4. Full RegimeEngine")

re = check("RegimeEngine(cfg)", lambda: RegimeEngine(cfg))
if re:
    result = check("RegimeEngine.run()", lambda: re.run(ohlcv))
    if result:
        print(f"       features shape : {result['features'].shape}")
        print(f"       regime_probs   : {list(result['regime_probs'].columns)}")
        mix_tail = result["mix"].tail(1)
        print(f"       mix (last bar) : {mix_tail.to_dict('records')}")

# ── 5. RegimeEngine with synthetic gamma injection ─────────────────────────────
section("5. RegimeEngine — gamma + L2 feature injection")

gamma_f = {
    "gamma_wall_distance": 0.03,
    "gamma_net":           1.5e8,
    "gamma_expiry_days":   5.0,
    "gex_flip_zone":       0.0,
}
l2_f = {
    "book_imbalance":  0.25,
    "ofi_true":        0.10,
    "spread_bps":      2.5,
    "depth_ratio":     1.1,
    "depth_trend":     0.5,
}

if re:
    result2 = check(
        "RegimeEngine.run(gamma+L2 injected)",
        lambda: re.run(ohlcv, gamma_features=gamma_f, l2_features=l2_f)
    )
    if result2:
        feat_cols = list(result2["features"].columns)
        extra = [c for c in feat_cols if c in list(gamma_f) + list(l2_f)]
        print(f"       injected columns found: {extra}")

# ── 6. PolygonRestClient (zero-signal graceful degradation) ────────────────────
section("6. Phase 3 — PolygonRestClient (no API key, graceful degradation)")

poly_cfg = {"enabled": False, "zero_on_missing": True}
check(
    "PolygonRestClient init (no key)",
    lambda: PolygonRestClient(poly_cfg)
)

# ─────────────────────────────────────────────────────────────────────────────
section("Summary")
if not failed:
    print("  ALL TESTS PASSED [OK]")
else:
    print(f"  {len(failed)} TEST(S) FAILED:")
    for lbl, err in failed:
        print(f"    • {lbl}: {err}")
    sys.exit(1)

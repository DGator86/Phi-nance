#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Dashboard — Full MFT Edition
----------------------------------------
All 6 tabs:
  1. ML Model Status   — model files + train buttons
  2. Fetch Data        — OHLCV download + training CSV
  3. MFT Blender       — REAL blending knobs from config.yaml,
                         auto-reactive, full pipeline stage view
  4. Phi-Bot           — universe scanner + Phi-Bot backtest
  5. Backtests         — classic strategy picker
  6. System Status     — engine health

Run:
    python -m streamlit run dashboard.py
"""

import copy
import io
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, date
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from engine_health import run_engine_health_check

# ── Lazy Lumibot / strategy loader ───────────────────────────────────────────
# Lumibot's credentials.py instantiates brokers at import time which requires
# env vars (TRADIER_TOKEN etc.).  We defer all imports until first use so
# that the dashboard can launch without any broker credentials configured.
_LUMIBOT_CACHE: dict = {}


def _strat(name: str):
    """Return a cached strategy class by module.ClassName string."""
    if name not in _LUMIBOT_CACHE:
        module_path, cls_name = name.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        _LUMIBOT_CACHE[name] = getattr(mod, cls_name)
    return _LUMIBOT_CACHE[name]


def _yahoo_backtesting():
    if "YahooDataBacktesting" not in _LUMIBOT_CACHE:
        from lumibot.backtesting import YahooDataBacktesting  # noqa: PLC0415
        _LUMIBOT_CACHE["YahooDataBacktesting"] = YahooDataBacktesting
    return _LUMIBOT_CACHE["YahooDataBacktesting"]


def _compute_accuracy(strat):
    from strategies.prediction_tracker import compute_prediction_accuracy  # noqa
    return compute_prediction_accuracy(strat)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join("regime_engine", "config.yaml")

REGIME_BINS = ["TREND_UP", "TREND_DN", "RANGE", "BREAKOUT_UP",
               "BREAKOUT_DN", "EXHAUST_REV", "LOWVOL", "HIGHVOL"]

REGIME_COLORS = {
    "TREND_UP":    "#22c55e", "TREND_DN":    "#ef4444",
    "RANGE":       "#94a3b8", "BREAKOUT_UP": "#f97316",
    "BREAKOUT_DN": "#a855f7", "EXHAUST_REV": "#ec4899",
    "LOWVOL":      "#06b6d4", "HIGHVOL":     "#eab308",
}

TAXONOMY_LEVELS = ["kingdom", "phylum", "class_", "order", "family", "genus"]

GATE_FEATURES = ["default", "d_mass_dt", "d_lambda", "mass",
                 "ofi_proxy", "dissipation_proxy"]

ML_MODELS = [
    ("Random Forest",       "models/classifier_rf.pkl",  "sklearn"),
    ("Gradient Boosting",   "models/classifier_gb.pkl",  "sklearn"),
    ("Logistic Regression", "models/classifier_lr.pkl",  "sklearn"),
    ("LightGBM",            "models/classifier_lgb.txt", "lgb"),
]

def _build_catalog() -> dict:
    """Build STRATEGY_CATALOG lazily so class imports happen on demand."""
    return {
        "Buy & Hold": {
            "module": "strategies.buy_and_hold.BuyAndHold",
            "description": "Naive long-only baseline.",
            "params": {"symbol": {"label": "Symbol", "type": "text", "default": "SPY"}},
        },
        "Momentum Rotation": {
            "module": "strategies.momentum.MomentumRotation",
            "description": "Strongest-momentum asset wins.",
            "params": {
                "symbols":        {"label": "Universe",      "type": "text",   "default": "SPY,VEU,AGG,GLD"},
                "lookback_days":  {"label": "Lookback (d)",  "type": "number", "default": 20, "min": 5, "max": 200},
                "rebalance_days": {"label": "Rebalance (d)", "type": "number", "default": 5,  "min": 1, "max": 60},
            },
        },
        "Mean Reversion (SMA)": {
            "module": "strategies.mean_reversion.MeanReversion",
            "description": "UP < SMA; DOWN > SMA.",
            "params": {
                "symbol":     {"label": "Symbol",     "type": "text",   "default": "SPY"},
                "sma_period": {"label": "SMA Period", "type": "number", "default": 20, "min": 5, "max": 200},
            },
        },
        "RSI": {
            "module": "strategies.rsi.RSIStrategy",
            "description": "UP < oversold; DOWN > overbought.",
            "params": {
                "symbol":     {"label": "Symbol",     "type": "text",   "default": "SPY"},
                "rsi_period": {"label": "RSI Period", "type": "number", "default": 14, "min": 2,  "max": 50},
                "oversold":   {"label": "Oversold",   "type": "number", "default": 30, "min": 10, "max": 50},
                "overbought": {"label": "Overbought", "type": "number", "default": 70, "min": 50, "max": 95},
            },
        },
        "Bollinger Bands": {
            "module": "strategies.bollinger.BollingerBands",
            "description": "UP below lower band; DOWN above upper.",
            "params": {
                "symbol":    {"label": "Symbol",   "type": "text",   "default": "SPY"},
                "bb_period": {"label": "BB Period", "type": "number", "default": 20, "min": 5, "max": 100},
                "num_std":   {"label": "Std Dev",   "type": "number", "default": 2,  "min": 1, "max": 4},
            },
        },
        "MACD": {
            "module": "strategies.macd.MACDStrategy",
            "description": "Bullish/bearish MACD crossover signal.",
            "params": {
                "symbol":        {"label": "Symbol",     "type": "text",   "default": "SPY"},
                "fast_period":   {"label": "Fast EMA",   "type": "number", "default": 12, "min": 2,  "max": 50},
                "slow_period":   {"label": "Slow EMA",   "type": "number", "default": 26, "min": 10, "max": 100},
                "signal_period": {"label": "Signal EMA", "type": "number", "default": 9,  "min": 2,  "max": 30},
            },
        },
        "Dual SMA Crossover": {
            "module": "strategies.dual_sma.DualSMACrossover",
            "description": "Golden cross / death cross.",
            "params": {
                "symbol":      {"label": "Symbol",   "type": "text",   "default": "SPY"},
                "fast_period": {"label": "Fast SMA", "type": "number", "default": 10, "min": 2,  "max": 100},
                "slow_period": {"label": "Slow SMA", "type": "number", "default": 50, "min": 10, "max": 300},
            },
        },
        "Channel Breakout": {
            "module": "strategies.breakout.ChannelBreakout",
            "description": "Donchian channel breakout/breakdown.",
            "params": {
                "symbol":         {"label": "Symbol",         "type": "text",   "default": "SPY"},
                "channel_period": {"label": "Channel Period", "type": "number", "default": 20, "min": 5, "max": 100},
            },
        },
        "Wyckoff": {
            "module": "strategies.wyckoff.WyckoffStrategy",
            "description": "Accumulation = UP; Distribution = DOWN.",
            "params": {
                "symbol":   {"label": "Symbol",         "type": "text",   "default": "SPY"},
                "lookback": {"label": "Lookback (days)", "type": "number", "default": 30, "min": 10, "max": 120},
            },
        },
        "Liquidity Pools": {
            "module": "strategies.liquidity_pools.LiquidityPoolStrategy",
            "description": "Resting liquidity + sweep reversals.",
            "params": {
                "symbol":         {"label": "Symbol",          "type": "text",   "default": "SPY"},
                "lookback":       {"label": "Lookback (days)",  "type": "number", "default": 40, "min": 15, "max": 120},
                "swing_strength": {"label": "Swing Strength",   "type": "number", "default": 3,  "min": 2,  "max": 10},
            },
        },
        "ML Classifier (RF)": {
            "module": "strategies.ml_classifier_strategy.MLClassifierStrategy",
            "description": "Random Forest on full MFT features.",
            "params": {
                "symbol":         {"label": "Symbol",   "type": "text",   "default": "SPY"},
                "min_confidence": {"label": "Min Conf", "type": "number", "default": 0.6, "min": 0.5, "max": 0.95},
            },
        },
        "ML Classifier (LightGBM)": {
            "module": "strategies.lightgbm_strategy.LightGBMStrategy",
            "description": "LightGBM on full MFT features.",
            "params": {
                "symbol":         {"label": "Symbol",   "type": "text",   "default": "SPY"},
                "min_confidence": {"label": "Min Conf", "type": "number", "default": 0.62, "min": 0.5, "max": 0.95},
            },
        },
        "Ensemble (RF+LGBM)": {
            "module": "strategies.ensemble_strategy.EnsembleMLStrategy",
            "description": "Both RF+LGBM must agree.",
            "params": {"symbol": {"label": "Symbol", "type": "text", "default": "SPY"}},
        },
        "Phi-Bot (MFT Default)": {
            "module": "strategies.blended_mft_strategy.BlendedMFTStrategy",
            "description": "Full MFT — default config weights.",
            "params": {
                "symbol":           {"label": "Symbol",     "type": "text",   "default": "SPY"},
                "signal_threshold": {"label": "Threshold",  "type": "number", "default": 0.15, "min": 0.05, "max": 0.50},
                "confidence_floor": {"label": "Conf Floor", "type": "number", "default": 0.30, "min": 0.10, "max": 0.80},
            },
        },
    }


STRATEGY_CATALOG: dict = {}  # filled lazily by _ensure_catalog()


def _ensure_catalog() -> dict:
    global STRATEGY_CATALOG
    if not STRATEGY_CATALOG:
        STRATEGY_CATALOG = _build_catalog()
    return STRATEGY_CATALOG


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_base_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_ohlcv(symbol: str, start, end) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        raw = yf.download(symbol, start=str(start), end=str(end),
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        return raw
    except Exception as e:
        st.error(f"Download failed for {symbol}: {e}")
        return None


def _run_engine_with_config(ohlcv: pd.DataFrame, cfg: dict) -> Optional[dict]:
    """Instantiate a fresh engine with the given config and run it."""
    try:
        from regime_engine.scanner import RegimeEngine
        engine = RegimeEngine(cfg)
        return engine.run(ohlcv)
    except Exception as e:
        st.error(f"Engine error: {e}")
        return None


@st.cache_resource
def _get_universe_scanner():
    from regime_engine.scanner import UniverseScanner
    return UniverseScanner(config_path=CONFIG_PATH)


def build_sidebar() -> dict:
    st.sidebar.title("Backtest Settings")
    s = st.sidebar.date_input("Start", value=date(2020, 1, 1), min_value=date(2005, 1, 1))
    e = st.sidebar.date_input("End",   value=date(2024, 12, 31), min_value=s)
    b = st.sidebar.number_input("Budget ($)", value=100_000, min_value=1_000, step=10_000)
    bm = st.sidebar.text_input("Benchmark", value="SPY")
    st.sidebar.markdown("---")
    st.sidebar.caption("Lumibot + Yahoo Finance")
    return {
        "start":     datetime.combine(s, datetime.min.time()),
        "end":       datetime.combine(e, datetime.min.time()),
        "budget":    float(b),
        "benchmark": bm,
    }


def _run_backtest(strategy_class, params, config):
    results, strat = strategy_class.run_backtest(
        datasource_class=_yahoo_backtesting(),
        backtesting_start=config["start"], backtesting_end=config["end"],
        budget=config["budget"], benchmark_asset=config["benchmark"],
        parameters=params,
        show_plot=False, show_tearsheet=False, save_tearsheet=False,
        show_indicators=False, show_progress_bar=False, quiet_logs=True,
    )
    return results, strat


# ─────────────────────────────────────────────────────────────────────────────
# Shared result displays
# ─────────────────────────────────────────────────────────────────────────────
def _show_portfolio(results, budget):
    if results is None:
        return
    try:
        def _g(obj, *keys):
            for k in keys:
                v = getattr(obj, k, None) or (obj.get(k) if hasattr(obj, "get") else None)
                if v is not None:
                    return v
            return None
        tr = _g(results, "total_return")
        cagr = _g(results, "cagr")
        dd = _g(results, "max_drawdown")
        sh = _g(results, "sharpe")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{tr:+.1%}" if tr is not None else "—")
        c2.metric("CAGR",         f"{cagr:+.1%}" if cagr is not None else "—")
        c3.metric("Max Drawdown", f"{dd:.1%}" if dd is not None else "—")
        c4.metric("Sharpe",       f"{sh:.2f}" if sh is not None else "—")
    except Exception:
        pass
    try:
        pv = getattr(results, "portfolio_value", None) or (
            results.get("portfolio_value") if hasattr(results, "get") else None)
        if pv is not None and len(pv) > 2:
            st.line_chart(pd.DataFrame(pv, columns=["Portfolio Value ($)"]),
                          use_container_width=True)
    except Exception:
        pass


def _show_accuracy(scorecard):
    if scorecard["total_predictions"] == 0:
        st.warning("No predictions recorded.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",    f"{scorecard['accuracy']:.1%}")
    c2.metric("Predictions", f"{scorecard['total_predictions']:,}")
    c3.metric("Correct",     f"{scorecard['hits']:,}")
    c4.metric("Wrong",       f"{scorecard['misses']:,}")
    cu, cd, ce = st.columns(3)
    cu.metric("UP Acc",   f"{scorecard['up_accuracy']:.1%}")
    cd.metric("DOWN Acc", f"{scorecard['down_accuracy']:.1%}")
    ce.metric("Edge",     f"${scorecard['edge']:.4f}")
    scored = scorecard["scored_log"]
    if len(scored) > 10:
        df = pd.DataFrame(scored)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["rolling"] = df["correct"].rolling(50, min_periods=10).mean()
        st.line_chart(df[["date", "rolling"]].dropna().set_index("date"),
                      y="rolling", use_container_width=True)
        st.caption("50-bar rolling prediction accuracy")
    with st.expander("Prediction log"):
        if scored:
            log = pd.DataFrame(scored)
            log["date"]        = pd.to_datetime(log["date"]).dt.date
            log["actual_move"] = log["actual_move"].map(lambda x: f"${x:+.2f}")
            log["correct"]     = log["correct"].map(lambda x: "OK" if x else "MISS")
            st.dataframe(
                log[["date", "symbol", "signal", "price",
                      "next_price", "actual_move", "correct"]],
                use_container_width=True, hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: ML Model Status
# ─────────────────────────────────────────────────────────────────────────────
def _check_sklearn(path):
    try:
        from regime_engine.ml_classifier import DirectionClassifier
        c = DirectionClassifier(); c.load(path)
        if not c.is_fitted: return False, None, None
        return True, getattr(c.model, "n_features_in_", "?"), list(getattr(c.model, "classes_", []))
    except Exception as e:
        return False, None, str(e)


def _check_lgb(path):
    try:
        from regime_engine.ml_classifier_lightgbm import LightGBMDirectionClassifier
        c = LightGBMDirectionClassifier(); c.load(path)
        if not c.is_fitted: return False, None, None
        return True, c.model.num_feature(), c.model.num_trees()
    except Exception as e:
        return False, None, str(e)


def render_ml_status():
    st.subheader("ML Model Files")
    cols = st.columns(len(ML_MODELS))
    any_missing = False
    for col, (display, path, kind) in zip(cols, ML_MODELS):
        with col:
            exists = os.path.exists(path)
            if not exists:
                any_missing = True
                st.error(f"**{display}**")
                st.caption(f"`{path}` — Missing")
            else:
                ok, n_feat, extra = _check_sklearn(path) if kind == "sklearn" else _check_lgb(path)
                kb = round(os.path.getsize(path) / 1024, 1)
                if ok:
                    st.success(f"**{display}**")
                    st.caption(f"`{path}` — {kb}KB")
                    if n_feat: st.caption(f"Features: {n_feat}")
                    if extra and kind == "lgb": st.caption(f"Trees: {extra}")
                else:
                    any_missing = True
                    st.warning(f"**{display}** — load error")

    st.markdown("---")
    csv = "historical_regime_features.csv"
    csv_ok = os.path.exists(csv)
    if csv_ok:
        try:
            n_rows = sum(1 for _ in open(csv, encoding="utf-8")) - 1
            n_cols = len(pd.read_csv(csv, nrows=1).columns)
            st.success(f"Training data ready: {n_rows:,} rows x {n_cols} cols — full MFT features")
        except Exception:
            st.warning("CSV exists but could not be read.")
    else:
        st.warning("No training CSV. Generate below or via the Fetch Data tab.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate MFT Training Data", disabled=csv_ok, use_container_width=True,
                     help="Downloads SPY 2015-2024, runs full MFT pipeline"):
            with st.status("Generating...", expanded=True) as s:
                r = subprocess.run([sys.executable, "generate_training_data.py"],
                                   capture_output=True, text=True)
                st.code(r.stdout + r.stderr)
                s.update(label="Done" if r.returncode == 0 else "Failed",
                         state="complete" if r.returncode == 0 else "error")
            st.rerun()
    with c2:
        if st.button("Train All Models", disabled=not csv_ok, type="primary",
                     use_container_width=True):
            with st.status("Training...", expanded=True) as s:
                r = subprocess.run([sys.executable, "train_ml_classifier.py"],
                                   capture_output=True, text=True)
                st.code(r.stdout + r.stderr)
                s.update(label="Done" if r.returncode == 0 else "Failed",
                         state="complete" if r.returncode == 0 else "error")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Fetch Data
# ─────────────────────────────────────────────────────────────────────────────
def render_fetch_data():
    st.subheader("Fetch & Preview OHLCV")
    ca, cb, cc = st.columns([2, 1, 1])
    with ca: traw = st.text_input("Tickers", value="SPY", key="fd_tickers")
    with cb: fs   = st.date_input("From", value=date(2018, 1, 1), key="fd_start")
    with cc: fe   = st.date_input("To",   value=date(2024, 12, 31), key="fd_end")
    tickers = [t.strip().upper() for t in traw.split(",") if t.strip()]

    c1, c2 = st.columns(2)
    fetch_clicked = c1.button("Fetch Data", type="primary", use_container_width=True)
    gen_clicked   = c2.button("Generate MFT Training CSV", use_container_width=True)

    if fetch_clicked and tickers:
        dfs = {}
        with st.spinner("Downloading..."):
            for sym in tickers:
                df = _load_ohlcv(sym, fs, fe)
                if df is not None:
                    dfs[sym] = df
        if dfs:
            st.session_state["fetched_data"] = dfs

    for sym, df in st.session_state.get("fetched_data", {}).items():
        st.markdown(f"#### {sym}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("From", str(df.index[0].date()))
        m3.metric("To",   str(df.index[-1].date()))
        if "close" in df.columns:
            ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
            m4.metric("Total Return", f"{ret:+.1%}")
            st.line_chart(df[["close"]].rename(columns={"close": f"{sym} Close"}))
        with st.expander("Last 20 rows"):
            st.dataframe(df.tail(20), use_container_width=True)

    if gen_clicked and tickers:
        primary = tickers[0]
        with st.status(f"Generating MFT CSV from {primary}...", expanded=True) as s:
            try:
                from regime_engine.scanner import RegimeEngine
                ohlcv = _load_ohlcv(primary, fs, fe)
                if ohlcv is None:
                    s.update(label="No data", state="error"); return
                cfg = _load_base_config()
                st.write(f"Running MFT pipeline on {len(ohlcv):,} bars...")
                engine = RegimeEngine(cfg)
                out    = engine.run(ohlcv)
                parts  = [
                    out["features"],
                    out["logits"].rename(columns=lambda c: f"logit_{c}"),
                    out["regime_probs"].rename(columns=lambda c: f"prob_{c}"),
                    out["mix"][[c for c in out["mix"].columns
                                if c in ("composite_signal","score","c_field",
                                         "c_consensus","c_liquidity")]],
                    out["signals"].rename(columns=lambda c: f"sig_{c}"),
                    out["weights"].rename(columns=lambda c: f"wt_{c}"),
                    out["projections"]["expected"].rename(columns=lambda c: f"proj_{c}"),
                ]
                combined = pd.concat([p.reindex(out["features"].index) for p in parts], axis=1)
                close = ohlcv["close"].values
                direction = np.zeros(len(close), dtype=int)
                direction[:-1] = (close[1:] > close[:-1]).astype(int)
                combined["direction"] = direction[:len(combined)]
                combined = combined.iloc[:-1].dropna()
                combined.to_csv("historical_regime_features.csv", index=False)
                s.update(label=f"Saved {len(combined):,} rows", state="complete")
                st.success("Now go to ML Model Status -> Train All Models")
                st.dataframe(combined.head(3), use_container_width=True)
            except Exception as e:
                s.update(label=f"Failed: {e}", state="error")
                st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: MFT Blender — REAL blending parameters
# ─────────────────────────────────────────────────────────────────────────────
def _blender_controls(base_cfg: dict, prefix: str) -> dict:
    """
    Render all real MFT blending function sliders.
    Returns a nested override dict to merge into base_cfg.
    """
    overrides: dict = {}

    # ── Section A: Taxonomy Smoothing (EWM alpha per level) ──────────────
    st.markdown("### A. Taxonomy Smoothing (EWM alpha per level)")
    st.caption(
        "Alpha = 1 - persistence. **Low alpha** = very sticky (slow to move). "
        "**High alpha** = fast response to new data."
    )
    cols = st.columns(len(TAXONOMY_LEVELS))
    tax_smooth = {}
    for col, level in zip(cols, TAXONOMY_LEVELS):
        with col:
            default = base_cfg["taxonomy"]["smoothing"].get(level, 0.15)
            val = st.slider(
                level.replace("_", ""), 0.01, 0.60, float(default), 0.01,
                key=f"{prefix}_smooth_{level}",
                help=f"Default: {default}"
            )
            tax_smooth[level] = val
    overrides.setdefault("taxonomy", {})["smoothing"] = tax_smooth

    # ── Section B: Gate Steepness γ per feature ───────────────────────────
    st.markdown("### B. Gate Steepness (gamma) per Feature")
    st.caption(
        "Controls how sharply the tanh gate responds to each feature. "
        "**Higher gamma** = hard threshold. **Lower** = smooth response."
    )
    cols = st.columns(len(GATE_FEATURES))
    gate_steep = {}
    for col, feat in zip(cols, GATE_FEATURES):
        with col:
            default = base_cfg["taxonomy"]["gate_steepness"].get(feat, 1.0)
            val = st.slider(
                feat, 0.1, 5.0, float(default), 0.1,
                key=f"{prefix}_gate_{feat}",
                help=f"Default: {default}"
            )
            gate_steep[feat] = val
    overrides["taxonomy"]["gate_steepness"] = gate_steep

    # ── Section C: MSL Condition Matrix scale ─────────────────────────────
    st.markdown("### C. MSL Kingdom Scale")
    st.caption(
        "Global multiplier on the MSL hardcoded condition matrix weights "
        "(Interface 2 & 3). 0.0 = disable MSL; 1.0 = full strength."
    )
    default_msl = float(base_cfg["taxonomy"].get("msl_kingdom_scale", 0.8))
    overrides["taxonomy"]["msl_kingdom_scale"] = st.slider(
        "msl_kingdom_scale", 0.0, 2.0, default_msl, 0.05,
        key=f"{prefix}_msl_scale",
        help=f"Default: {default_msl}"
    )

    # ── Section D: Mixer / Confidence blending ────────────────────────────
    st.markdown("### D. Mixer / Confidence Blending (Interface 4, 6)")
    st.caption(
        "**affinity_blend**: 0 = pure validity weights; 1 = pure affinity matrix. "
        "**interaction_alpha**: 1.0 = pure linear composite; lower = more cross-indicator interaction."
    )
    conf_cfg = base_cfg["confidence"]
    d1, d2, d3, d4 = st.columns(4)
    conf_overrides: dict = {}
    with d1:
        conf_overrides["affinity_blend"] = st.slider(
            "affinity_blend", 0.0, 1.0,
            float(conf_cfg.get("affinity_blend", 0.5)), 0.05,
            key=f"{prefix}_affinity",
            help="Interface 4: validity vs affinity matrix blend"
        )
    with d2:
        conf_overrides["interaction_alpha"] = st.slider(
            "interaction_alpha", 0.0, 1.0,
            float(conf_cfg.get("interaction_alpha", 0.7)), 0.05,
            key=f"{prefix}_interaction",
            help="Interface 6: linear vs quadratic composite blend"
        )
    with d3:
        conf_overrides["liquidity_volume_scale"] = st.slider(
            "vol_scale", 0.1, 2.0,
            float(conf_cfg.get("liquidity_volume_scale", 0.5)), 0.05,
            key=f"{prefix}_vol_scale",
            help="sigmoid(vol_zscore * scale) — liquidity confidence"
        )
    with d4:
        conf_overrides["liquidity_gap_scale"] = st.slider(
            "gap_scale", 0.1, 2.0,
            float(conf_cfg.get("liquidity_gap_scale", 0.5)), 0.05,
            key=f"{prefix}_gap_scale",
            help="sigmoid(-|gap| * scale) — gap penalty on liquidity confidence"
        )
    overrides["confidence"] = conf_overrides

    # ── Section E: Projection AR(1) per-regime parameters ─────────────────
    st.markdown("### E. Projection AR(1) Parameters per Regime")
    st.caption(
        "**mu**: equilibrium mean. **phi**: persistence (autocorrelation). "
        "**beta**: momentum coefficient. **sigma**: residual noise std."
    )
    proj_cfg = base_cfg["projection"]["regimes"]
    proj_overrides: dict = {"regimes": {}}

    for regime in REGIME_BINS:
        rp = proj_cfg.get(regime, {})
        with st.expander(f"{regime}  (mu={rp.get('mu',0):+.2f}, phi={rp.get('phi',0):.2f})"):
            r1, r2, r3, r4 = st.columns(4)
            proj_overrides["regimes"][regime] = {
                "mu":    r1.slider("mu",    -1.0, 1.0, float(rp.get("mu",   0.0)), 0.05, key=f"{prefix}_proj_{regime}_mu"),
                "phi":   r2.slider("phi",   -1.0, 1.0, float(rp.get("phi",  0.3)), 0.05, key=f"{prefix}_proj_{regime}_phi"),
                "beta":  r3.slider("beta",  -1.0, 1.0, float(rp.get("beta", 0.0)), 0.05, key=f"{prefix}_proj_{regime}_beta"),
                "sigma": r4.slider("sigma",  0.01, 1.0, float(rp.get("sigma",0.2)), 0.01, key=f"{prefix}_proj_{regime}_sigma"),
            }
    overrides["projection"] = proj_overrides

    return overrides


def _run_and_display_pipeline(ohlcv: pd.DataFrame, cfg: dict, sym: str):
    """Run the full engine and display every pipeline stage."""
    with st.spinner("Running full MFT pipeline..."):
        out = _run_engine_with_config(ohlcv, cfg)
    if out is None:
        return

    feat_df      = out["features"]
    logits_df    = out["logits"]
    regime_df    = out["regime_probs"]
    mix_df       = out["mix"]
    signals_df   = out["signals"]
    weights_df   = out["weights"]
    proj_exp     = out["projections"]["expected"]
    proj_var     = out["projections"]["variance"]

    # Store for live-refresh access
    st.session_state["blender_out"]  = out
    st.session_state["blender_ohlcv"] = ohlcv
    st.session_state["blender_cfg"]  = cfg

    # ── Stage 1: Features ─────────────────────────────────────────────────
    with st.expander("STAGE 1 — Feature Engine Output", expanded=False):
        feat_cols = list(feat_df.columns)
        sel_feats = st.multiselect("Show features", feat_cols,
                                   default=feat_cols[:6], key="blender_feat_sel")
        if sel_feats:
            st.line_chart(feat_df[sel_feats].tail(252), use_container_width=True)
        st.caption(f"{len(feat_cols)} computed features")

    # ── Stage 2: Taxonomy Logits ──────────────────────────────────────────
    with st.expander("STAGE 2 — Taxonomy Logits (Kingdom → Genus)", expanded=True):
        node_groups = {
            "Kingdom (DIR, NDR, TRN)":       [c for c in logits_df.columns if c in ("DIR","NDR","TRN")],
            "Phylum (LV, NV, HV)":           [c for c in logits_df.columns if c in ("LV","NV","HV")],
            "Class (PT,PX,TE,BR,RR,AR,SR,RB,FB)": [c for c in logits_df.columns
                                                     if c in ("PT","PX","TE","BR","RR","AR","SR","RB","FB")],
            "Order":                         [c for c in logits_df.columns if c in ("AGC","RVP","ABS","EXH")],
            "Family":                        [c for c in logits_df.columns if c in ("ALN","CT","CST")],
            "Genus":                         [c for c in logits_df.columns if c in ("RUN","PBM","FLG","VWM","RRO","SRR")],
        }
        for grp_name, grp_cols in node_groups.items():
            if grp_cols:
                st.markdown(f"**{grp_name}**")
                st.line_chart(logits_df[grp_cols].tail(252), use_container_width=True)

    # ── Stage 3: Regime Probabilities ─────────────────────────────────────
    with st.expander("STAGE 3 — Probability Field (Regime Probabilities)", expanded=True):
        st.area_chart(regime_df.tail(252), use_container_width=True)
        st.caption("8-regime probability field — stacked area shows regime mix over time")

        # Latest bar breakdown
        latest = regime_df.iloc[-1].sort_values(ascending=False)
        st.markdown("**Latest bar:**")
        regime_cols = st.columns(4)
        for i, (r, p) in enumerate(latest.items()):
            regime_cols[i % 4].metric(r, f"{p:.1%}")

    # ── Stage 4: Indicator Signals + Validity Weights ─────────────────────
    with st.expander("STAGE 4 — Indicator Signals & Validity Weights", expanded=False):
        sa, sb = st.tabs(["Signals", "Validity Weights"])
        with sa:
            st.line_chart(signals_df.tail(252), use_container_width=True)
            st.caption("Normalized indicator signals (z-scored)")
        with sb:
            st.line_chart(weights_df.tail(252), use_container_width=True)
            st.caption("Per-indicator validity weights (affinity-blended)")

    # ── Stage 5: Mixer Output ─────────────────────────────────────────────
    with st.expander("STAGE 5 — Mixer / Composite Score", expanded=True):
        conf_cols = [c for c in mix_df.columns
                     if c in ("composite_signal","score","c_field","c_consensus","c_liquidity")]
        if conf_cols:
            st.line_chart(mix_df[conf_cols].tail(252), use_container_width=True)
        last_mix = mix_df.iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Composite",   f"{last_mix.get('composite_signal', 0):+.3f}")
        m2.metric("Score",       f"{last_mix.get('score', 0):+.3f}")
        m3.metric("C_field",     f"{last_mix.get('c_field', 0):.3f}")
        m4.metric("C_consensus", f"{last_mix.get('c_consensus', 0):.3f}")
        m5.metric("C_liquidity", f"{last_mix.get('c_liquidity', 0):.3f}")
        overall = (last_mix.get("c_field", 0) *
                   last_mix.get("c_consensus", 0) *
                   last_mix.get("c_liquidity", 0))
        if overall >= 0.1:
            st.success(f"Overall confidence: {overall:.3f} — signal is tradeable")
        else:
            st.warning(f"Overall confidence: {overall:.3f} — below trade threshold")

    # ── Stage 6: Projections ──────────────────────────────────────────────
    with st.expander("STAGE 6 — Projection Engine (AR(1) Expected Values)", expanded=False):
        pe, pv = st.tabs(["Expected Value", "Variance"])
        with pe:
            st.line_chart(proj_exp.tail(252), use_container_width=True)
            st.caption("Per-indicator regime-weighted AR(1) expected next value")
        with pv:
            st.line_chart(proj_var.tail(252), use_container_width=True)
            st.caption("Per-indicator mixture variance (uncertainty)")


def render_mft_blender(config: dict):
    st.subheader("MFT Blender — Real Configuration Parameters")
    st.caption(
        "All sliders map directly to `regime_engine/config.yaml`. "
        "Changes propagate through the full pipeline (Taxonomy smoothing → Gate steepness → "
        "MSL condition matrix → Mixer blend functions → Projection AR(1))."
    )

    base_cfg = _load_base_config()

    # ── Ticker + Auto-refresh controls ────────────────────────────────────
    col_sym, col_s, col_e, col_auto = st.columns([2, 1, 1, 1])
    with col_sym:   sym       = st.text_input("Symbol", value="SPY", key="blender_sym")
    with col_s:     bl_start  = st.date_input("From", value=date(2020, 1, 1), key="blender_start")
    with col_e:     bl_end    = st.date_input("To",   value=date(2024, 12, 31), key="blender_end")
    with col_auto:
        auto_refresh = st.toggle("Auto-refresh", value=False, key="blender_auto")
        refresh_secs = st.number_input("Interval (s)", value=30, min_value=5, max_value=300,
                                       key="blender_interval") if auto_refresh else None

    # ── Real blending parameter controls ─────────────────────────────────
    with st.form("blender_form"):
        overrides = _blender_controls(base_cfg, "bl")
        submitted = st.form_submit_button(
            "Run Pipeline with These Parameters",
            type="primary",
            use_container_width=True,
        )

    if submitted or (auto_refresh and st.session_state.get("blender_auto_ran")):
        ohlcv = _load_ohlcv(sym, bl_start, bl_end)
        if ohlcv is None:
            return
        merged_cfg = _deep_merge(base_cfg, overrides)
        _run_and_display_pipeline(ohlcv, merged_cfg, sym)
        st.session_state["blender_auto_ran"] = True

    # ── Auto-refresh trigger ──────────────────────────────────────────────
    if auto_refresh and refresh_secs and st.session_state.get("blender_auto_ran"):
        st.caption(f"Auto-refresh every {refresh_secs}s — last run: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(int(refresh_secs))
        st.rerun()
    elif not auto_refresh and not st.session_state.get("blender_auto_ran"):
        st.info(
            "Adjust any blending parameter above, then click "
            "**Run Pipeline with These Parameters** to see the full pipeline output."
        )

    # ── Show last results if available ───────────────────────────────────
    if not submitted and st.session_state.get("blender_out") is not None:
        st.caption("Showing last run results:")
        out   = st.session_state["blender_out"]
        ohlcv = st.session_state["blender_ohlcv"]
        cfg   = st.session_state.get("blender_cfg", base_cfg)
        _run_and_display_pipeline.__wrapped__ = True  # prevent double-run
        # Re-display from cached output
        with st.expander("STAGE 2 — Taxonomy Logits", expanded=True):
            logits_df = out["logits"]
            k_cols = [c for c in logits_df.columns if c in ("DIR","NDR","TRN")]
            if k_cols:
                st.markdown("**Kingdom**")
                st.line_chart(logits_df[k_cols].tail(252), use_container_width=True)
        with st.expander("STAGE 3 — Regime Probabilities", expanded=True):
            st.area_chart(out["regime_probs"].tail(252), use_container_width=True)
        with st.expander("STAGE 5 — Mixer / Composite Score", expanded=True):
            mix_df = out["mix"]
            conf_cols = [c for c in mix_df.columns
                         if c in ("composite_signal","score","c_field","c_consensus","c_liquidity")]
            if conf_cols:
                st.line_chart(mix_df[conf_cols].tail(252), use_container_width=True)

    # ── Blended backtest ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Backtest with Current Blend")
    if st.button("Run Blended Backtest", use_container_width=True):
        if st.session_state.get("blender_out") is None:
            st.warning("Run the pipeline first.")
        else:
            merged_cfg = _deep_merge(base_cfg, overrides if submitted else {})
            with st.status("Running backtest...", expanded=True) as s:
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        _BlendedMFT = _strat(
                            "strategies.blended_mft_strategy.BlendedMFTStrategy"
                        )
                        results, strat = _run_backtest(
                            _BlendedMFT,
                            {"symbol": sym, "indicator_weights": {}},
                            config,
                        )
                    sc = _compute_accuracy(strat)
                    s.update(label=f"Done — {sc['accuracy']:.1%}", state="complete")
                    bt1, bt2 = st.tabs(["Portfolio", "Accuracy"])
                    with bt1: _show_portfolio(results, config["budget"])
                    with bt2: _show_accuracy(sc)
                except Exception as e:
                    s.update(label=f"Failed: {e}", state="error")
                    st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Phi-Bot
# ─────────────────────────────────────────────────────────────────────────────
def render_phi_bot(config: dict):
    st.subheader("Phi-Bot — Full MFT System")
    st.caption("Pure MFT — no manual tuning. The full pipeline drives every signal.")

    scan_tab, bt_tab = st.tabs(["Regime Scanner", "Phi-Bot Backtest"])

    with scan_tab:
        st.markdown("#### Universe Regime Scanner")
        ca, cb, cc = st.columns([3, 1, 1])
        with ca: universe_raw = st.text_input(
            "Universe", value="SPY, QQQ, AAPL, NVDA, TSLA, MSFT, AMZN, GLD",
            key="phi_universe"
        )
        with cb: scan_start = st.date_input("From", value=date(2022, 1, 1), key="phi_scan_start")
        with cc: scan_end   = st.date_input("To",   value=date(2024, 12, 31), key="phi_scan_end")

        col_sort, col_auto = st.columns([2, 1])
        sort_col  = col_sort.selectbox("Sort by", ["score","composite_signal","c_field","c_consensus"],
                                       key="phi_sort")
        live_scan = col_auto.toggle("Live scan (auto-refresh)", key="phi_live")
        if live_scan:
            refresh_s = st.slider("Refresh every (s)", 15, 300, 60, key="phi_refresh_s")

        if st.button("Scan Universe", type="primary", use_container_width=True) or live_scan:
            tickers = [t.strip().upper() for t in universe_raw.split(",") if t.strip()]
            with st.status(f"Scanning {len(tickers)} tickers...", expanded=True) as s:
                universe: Dict[str, pd.DataFrame] = {}
                for sym in tickers:
                    st.write(f"  {sym}...")
                    df = _load_ohlcv(sym, scan_start, scan_end)
                    if df is not None:
                        universe[sym] = df
                if not universe:
                    s.update(label="No data", state="error")
                else:
                    try:
                        scanner = _get_universe_scanner()
                        results_df = scanner.scan(universe, sort_by=sort_col)
                        st.session_state["scan_results"] = results_df
                        st.session_state["scan_ts"] = datetime.now().strftime("%H:%M:%S")
                        s.update(label=f"Scanned {len(results_df)} tickers", state="complete")
                    except Exception as e:
                        s.update(label=f"Failed: {e}", state="error")
                        st.exception(e)

        scan_df = st.session_state.get("scan_results")
        if scan_df is not None and not scan_df.empty:
            ts = st.session_state.get("scan_ts", "")
            st.caption(f"Last scan: {ts}")

            # Color-coded table
            display_cols = [c for c in ["ticker","score","composite_signal",
                                        "c_field","c_consensus","c_liquidity",
                                        "top_species","top_species_desc"]
                            if c in scan_df.columns]
            st.dataframe(scan_df[display_cols], use_container_width=True, hide_index=True)

            # Regime prob bars per ticker
            prob_cols = [c for c in scan_df.columns if c.startswith("p_")]
            if prob_cols:
                st.markdown("**Regime probabilities by ticker**")
                prob_df = (scan_df.set_index("ticker")[prob_cols]
                           .rename(columns=lambda c: c.replace("p_","")))
                st.bar_chart(prob_df.T, use_container_width=True)

        if live_scan and st.session_state.get("scan_results") is not None:
            st.caption(f"Next refresh in {refresh_s}s...")
            time.sleep(refresh_s)
            st.rerun()

    with bt_tab:
        st.markdown("#### Phi-Bot Backtest")
        bt_sym = st.text_input("Symbol", value="SPY", key="phi_bt_sym")
        ct, cc2 = st.columns(2)
        threshold  = ct.slider("Signal threshold", 0.05, 0.50, 0.15, 0.05, key="phi_thresh")
        conf_floor = cc2.slider("Confidence floor", 0.10, 0.80, 0.30, 0.05, key="phi_conf")

        if st.button("Run Phi-Bot Backtest", type="primary", use_container_width=True):
            with st.status("Running Phi-Bot...", expanded=True) as s:
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        _BlendedMFT = _strat(
                            "strategies.blended_mft_strategy.BlendedMFTStrategy"
                        )
                        results, strat = _run_backtest(
                            _BlendedMFT,
                            {"symbol": bt_sym, "indicator_weights": {},
                             "signal_threshold": threshold,
                             "confidence_floor": conf_floor},
                            config,
                        )
                    sc = _compute_accuracy(strat)
                    s.update(label=f"Phi-Bot — {sc['accuracy']:.1%}", state="complete")
                    t1, t2 = st.tabs(["Portfolio", "Accuracy"])
                    with t1: _show_portfolio(results, config["budget"])
                    with t2: _show_accuracy(sc)
                except Exception as e:
                    s.update(label=f"Failed: {e}", state="error")
                    st.exception(e)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: Classic Backtests
# ─────────────────────────────────────────────────────────────────────────────
def _strategy_card(name, info):
    enabled = st.toggle(f"**{name}**", key=f"t_{name}")
    if enabled:
        st.caption(info["description"])
        params = {}
        for key, spec in info["params"].items():
            wk = f"{name}_{key}"
            if spec["type"] == "text":
                params[key] = st.text_input(spec["label"], value=spec["default"], key=wk)
            elif spec["type"] == "number":
                params[key] = st.number_input(spec["label"], value=spec["default"],
                                              min_value=spec.get("min", 1),
                                              max_value=spec.get("max", 9999), key=wk)
        return True, params
    st.caption(info["description"])
    return False, None


def _resolve(name, raw):
    resolved = dict(raw)
    if name == "Momentum Rotation" and "symbols" in resolved:
        resolved["symbols"] = [s.strip() for s in resolved["symbols"].split(",") if s.strip()]
    for k in ("lookback_days","rebalance_days","sma_period","rsi_period","oversold",
              "overbought","bb_period","fast_period","slow_period","signal_period",
              "channel_period","lookback","swing_strength"):
        if k in resolved:
            resolved[k] = int(resolved[k])
    return resolved


def render_backtests(config: dict):
    st.subheader("Select Strategies to Backtest")
    selected = {}
    items = list(_ensure_catalog().items())
    for row_start in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (name, info) in zip(cols, items[row_start:row_start + 4]):
            with col:
                with st.container(border=True):
                    enabled, params = _strategy_card(name, info)
                    if enabled and params is not None:
                        selected[name] = {
                            "cls": _strat(info["module"]),
                            "params": _resolve(name, params),
                        }

    st.markdown("---")
    if not selected:
        st.info("Enable at least one strategy.")
        return

    st.caption(f"Ready: {', '.join(selected.keys())}")
    if st.button("Run Backtests", type="primary", use_container_width=True):
        all_sc, all_res = {}, {}
        for name, entry in selected.items():
            with st.status(f"Running {name}...", expanded=True) as s:
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        results, strat = _run_backtest(
                            entry["cls"], entry["params"], config
                        )
                    sc = _compute_accuracy(strat)
                    all_sc[name] = sc
                    all_res[name] = results
                    s.update(label=f"{name} — {sc['accuracy']:.1%}", state="complete")
                except Exception as e:
                    s.update(label=f"{name} — failed", state="error")
                    st.error(str(e))

        for name, sc in all_sc.items():
            st.markdown(f"### {name}")
            t1, t2 = st.tabs(["Portfolio", "Accuracy"])
            with t1: _show_portfolio(all_res.get(name), config["budget"])
            with t2: _show_accuracy(sc)

        if len(all_sc) > 1:
            st.markdown("---")
            st.subheader("Comparison")
            rows = [{"Strategy": n, "Accuracy": f"{sc['accuracy']:.1%}",
                     "Predictions": sc["total_predictions"],
                     "Edge": f"${sc['edge']:.4f}"}
                    for n, sc in all_sc.items() if sc["total_predictions"] > 0]
            rows.sort(key=lambda r: float(r["Accuracy"].strip("%")) / 100, reverse=True)
            st.table(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6: System Status
# ─────────────────────────────────────────────────────────────────────────────
def render_system_status():
    st.subheader("Regime Engine Health Check")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("Check engine", type="primary") or c2.button("Re-check") or \
       st.session_state.get("engine_health") is None:
        with st.status("Running health check...", expanded=True) as s:
            try:
                health = run_engine_health_check()
                st.session_state["engine_health"] = health
                s.update(label="Complete" if not health.get("error") else "Failed",
                         state="complete" if not health.get("error") else "error")
            except Exception as e:
                st.session_state["engine_health"] = {"ok": False, "error": str(e),
                                                      "components": {}, "optional": {}}
                s.update(label="Error", state="error")

    health = st.session_state.get("engine_health")
    if health is None:
        return
    if health.get("error") and not health.get("components"):
        st.error(health["error"]); return

    if health.get("ok"):
        st.success("All MFT pipeline components connected.")
    else:
        st.warning("One or more components failed.")

    cols = st.columns(3)
    for idx, (name, c) in enumerate(health.get("components", {}).items()):
        with cols[idx % 3]:
            st.markdown(f"**{name}** — {'OK' if c.get('ok') else 'FAIL'}")
            st.caption(c.get("message", ""))

    with st.expander("Optional"):
        for k, v in health.get("optional", {}).items():
            st.text(f"  {k}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Phi-nance", page_icon="📊", layout="wide")
    st.title("Phi-nance — Market Field Theory Dashboard")
    st.caption("Real MFT blending parameters · Full pipeline view · Live scanner · Phi-Bot")

    config = build_sidebar()

    tabs = st.tabs([
        "ML Model Status",
        "Fetch Data",
        "MFT Blender",
        "Phi-Bot",
        "Backtests",
        "System Status",
    ])

    with tabs[0]: render_ml_status()
    with tabs[1]: render_fetch_data()
    with tabs[2]: render_mft_blender(config)
    with tabs[3]: render_phi_bot(config)
    with tabs[4]: render_backtests(config)
    with tabs[5]: render_system_status()


if __name__ == "__main__":
    main()

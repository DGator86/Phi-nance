#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Dashboard -- Premium $250/mo SaaS Edition v3.1
=========================================================
Production-quality quant trading dashboard with:
  - AUTO WEB/MOBILE DETECTION with adaptive layout
  - Animated hero landing with live stats
  - Glassmorphism dark UI with purple/orange neural theme
  - Real-time regime detection visualizations
  - Professional portfolio analytics
  - Premium Plotly charts with animated transitions
  - Feature showcase & pricing display
  - Dynamic sidebar with engine status
  - Mobile: bottom nav, stacked columns, touch-friendly
  - Tablet: hybrid 2-3 column layout

Run:
    python -m streamlit run dashboard.py
"""

import copy, io, os, subprocess, sys, time, json, hashlib
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, date, timedelta
from typing import Dict, Optional
from pathlib import Path

os.environ.setdefault("IS_BACKTESTING", "True")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (BrokenPipeError, OSError):
    pass  # pipe already closed during Streamlit hot-reload

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

from engine_health import run_engine_health_check
from app_streamlit.device_detect import detect_device, get_device

# ---------------------------------------------------------------------------
# Premium CSS injection
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_CSS_PATH = _ROOT / ".streamlit" / "styles.css"


def _inject_premium_css():
    """Inject premium CSS via st.markdown and JS/fonts via components.html.
    
    st.markdown strips <script> and <link> tags even with unsafe_allow_html.
    Use components.html (zero-height iframe) for those.
    """
    import streamlit.components.v1 as components
    from app_streamlit.device_detect import _JS_DETECT

    css = _CSS_PATH.read_text(encoding="utf-8") if _CSS_PATH.exists() else ""

    # CSS injection (st.markdown handles <style> fine)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # JS + font loading via zero-height iframe (st.markdown strips <script>/<link>)
    components.html(_JS_DETECT, height=0)


# ---------------------------------------------------------------------------
# Mobile Components
# ---------------------------------------------------------------------------
def _render_mobile_nav():
    """Bottom navigation bar is disabled — Streamlit tabs handle navigation."""
    return


def _render_mobile_header():
    """Render compact mobile header bar."""
    dev = get_device()
    if dev.is_desktop:
        return
    st.markdown(f"""
    <div class="phi-mobile-header">
        <div class="phi-mobile-header-brand">PHI-NANCE</div>
        <div class="phi-mobile-header-status">
            {_render_status_dot("live")} LIVE &middot; {datetime.now().strftime("%H:%M")}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Premium HTML Components
# ---------------------------------------------------------------------------
def _premium_sidebar_brand():
    """Render the premium sidebar branding with version badge."""
    st.sidebar.markdown("""
    <div class="phi-logo-container">
        <div class="phi-logo-text">PHI-NANCE</div>
        <div class="phi-logo-sub">MARKET FIELD THEORY</div>
        <div class="phi-logo-version">v3.0 PREMIUM</div>
    </div>
    """, unsafe_allow_html=True)


def _render_signal_badge(signal: str) -> str:
    cls = {"BUY": "phi-signal-buy", "SELL": "phi-signal-sell"}.get(signal, "phi-signal-hold")
    return f'<span class="phi-signal {cls}">{signal}</span>'


def _render_status_dot(status: str) -> str:
    return f'<span class="phi-status-dot {status}"></span>'


def _render_kpi_row(kpis: list) -> str:
    """Render a row of animated KPI cards. Each kpi = (label, value, delta, delta_type)."""
    cards = ""
    for label, value, delta, dtype in kpis:
        delta_html = f'<div class="phi-kpi-delta {dtype}">{delta}</div>' if delta else ""
        cards += f"""
        <div class="phi-kpi-card">
            <div class="phi-kpi-label">{label}</div>
            <div class="phi-kpi-value">{value}</div>
            {delta_html}
        </div>"""
    return f'<div class="phi-kpi-row">{cards}</div>'


def _render_section_header(icon: str, title: str, badge: str = "") -> str:
    badge_html = f'<span class="phi-section-badge">{badge}</span>' if badge else ""
    return f"""
    <div class="phi-section-header">
        <span class="phi-section-icon">{icon}</span>
        <span class="phi-section-title">{title}</span>
        {badge_html}
    </div>"""


def _render_hero():
    """Render the premium hero section -- mobile-adaptive."""
    dev = get_device()
    if dev.is_phone:
        # Compact phone hero
        st.markdown("""
        <div class="phi-hero">
            <div class="phi-hero-badge">QUANT PLATFORM</div>
            <div class="phi-hero-title">Phi-nance</div>
            <div class="phi-hero-subtitle">
                AI-powered regime detection, backtesting &amp; signal generation.
            </div>
            <div class="phi-hero-stats">
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">15+</div>
                    <div class="phi-hero-stat-label">Strategies</div>
                </div>
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">8</div>
                    <div class="phi-hero-stat-label">Regimes</div>
                </div>
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">6</div>
                    <div class="phi-hero-stat-label">ML</div>
                </div>
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">24/7</div>
                    <div class="phi-hero-stat-label">Live</div>
                </div>
            </div>
        </div>
        <div class="phi-gradient-bar"></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="phi-hero">
            <div class="phi-hero-badge">QUANTITATIVE RESEARCH PLATFORM</div>
            <div class="phi-hero-title">Phi-nance</div>
            <div class="phi-hero-subtitle">
                AI-powered Market Field Theory engine for institutional-grade regime detection,
                multi-strategy backtesting, and real-time signal generation.
            </div>
            <div class="phi-hero-stats">
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">15+</div>
                    <div class="phi-hero-stat-label">Strategies</div>
                </div>
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">8</div>
                    <div class="phi-hero-stat-label">Regime States</div>
                </div>
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">6</div>
                    <div class="phi-hero-stat-label">ML Models</div>
                </div>
                <div class="phi-hero-stat">
                    <div class="phi-hero-stat-value">24/7</div>
                    <div class="phi-hero-stat-label">Live Scanner</div>
                </div>
            </div>
        </div>
        <div class="phi-gradient-bar"></div>
        """, unsafe_allow_html=True)


def _render_features():
    """Render the feature showcase grid."""
    st.markdown("""
    <div class="phi-features-grid">
        <div class="phi-feature-card">
            <div class="phi-feature-icon">&#x26A1;</div>
            <div class="phi-feature-title">Market Field Theory</div>
            <div class="phi-feature-desc">8-regime probability field with taxonomy-driven signal decomposition. Real-time composite scoring.</div>
        </div>
        <div class="phi-feature-card">
            <div class="phi-feature-icon">&#x1F9E0;</div>
            <div class="phi-feature-title">AI Signal Engine</div>
            <div class="phi-feature-desc">Random Forest, LightGBM, and ensemble models trained on full MFT feature vectors.</div>
        </div>
        <div class="phi-feature-card">
            <div class="phi-feature-icon">&#x1F4CA;</div>
            <div class="phi-feature-title">Multi-Strategy Arena</div>
            <div class="phi-feature-desc">15+ strategies from RSI to Wyckoff, head-to-head comparison with Lumibot integration.</div>
        </div>
        <div class="phi-feature-card">
            <div class="phi-feature-icon">&#x1F916;</div>
            <div class="phi-feature-title">Plutus LLM Bot</div>
            <div class="phi-feature-desc">0xroyce/Plutus -- LLaMA 3.1-8B fine-tuned on 394 finance books with in-context learning.</div>
        </div>
        <div class="phi-feature-card">
            <div class="phi-feature-icon">&#x1F4B0;</div>
            <div class="phi-feature-title">Options Analytics</div>
            <div class="phi-feature-desc">Delta-based simulation with Black-Scholes pricing, IV surface analysis, and P&L tracking.</div>
        </div>
        <div class="phi-feature-card">
            <div class="phi-feature-icon">&#x1F30D;</div>
            <div class="phi-feature-title">Universe Scanner</div>
            <div class="phi-feature-desc">Scan entire watchlists through the MFT pipeline. Regime heatmaps and ranked signals.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_footer():
    dev = get_device()
    # Extra padding on mobile for bottom nav bar
    extra_pad = "padding-bottom:4rem;" if dev.is_mobile else ""
    st.markdown(f"""
    <div class="phi-footer" style="{extra_pad}">
        PHI-NANCE &middot; MARKET FIELD THEORY ENGINE &middot; v3.1 PREMIUM
        <br>QUANTITATIVE RESEARCH PLATFORM &middot; {datetime.now().strftime("%Y")}
        <br><span style="font-size:0.55rem;opacity:0.5;">
            {dev.device_type.value.upper()} MODE &middot; {dev.screen_width}px
        </span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly Premium Theme
# ---------------------------------------------------------------------------
PHI_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,18,26,0.3)",
    font=dict(color="#eeeef2", family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor="rgba(168,85,247,0.05)", zerolinecolor="rgba(168,85,247,0.08)", showgrid=True),
    yaxis=dict(gridcolor="rgba(168,85,247,0.05)", zerolinecolor="rgba(168,85,247,0.08)", showgrid=True),
    legend=dict(bgcolor="rgba(18,18,26,0.7)", bordercolor="rgba(168,85,247,0.12)", borderwidth=1, font=dict(size=11)),
    hoverlabel=dict(bgcolor="#1a1a26", bordercolor="#a855f7", font=dict(color="#eeeef2", size=12)),
)

CHART_COLORS = ["#a855f7", "#f97316", "#22c55e", "#06b6d4", "#eab308", "#ec4899", "#8b5cf6", "#14b8a6"]

REGIME_COLORS = {
    "TREND_UP": "#22c55e", "TREND_DN": "#ef4444",
    "RANGE": "#94a3b8", "BREAKOUT_UP": "#f97316",
    "BREAKOUT_DN": "#a855f7", "EXHAUST_REV": "#ec4899",
    "LOWVOL": "#06b6d4", "HIGHVOL": "#eab308",
}


def _phi_chart(fig: go.Figure, height: int = 0) -> None:
    """Render a Plotly chart with device-aware height and touch settings."""
    dev = get_device()
    if height <= 0:
        height = dev.chart_height
    elif dev.is_phone:
        height = max(220, int(height * 0.7))
    elif dev.is_tablet:
        height = max(260, int(height * 0.85))
    fig.update_layout(**PHI_PLOTLY_LAYOUT, height=height, xaxis_rangeslider_visible=False)
    show_modebar = dev.is_desktop
    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": show_modebar, "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "scrollZoom": not dev.is_mobile,
    })


# ---------------------------------------------------------------------------
# Lazy Imports (Lumibot / strategies)
# ---------------------------------------------------------------------------
_LUMIBOT_CACHE: dict = {}


def _strat(name: str):
    if name not in _LUMIBOT_CACHE:
        os.environ["IS_BACKTESTING"] = "True"
        module_path, cls_name = name.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        _LUMIBOT_CACHE[name] = getattr(mod, cls_name)
    return _LUMIBOT_CACHE[name]


def _av_backtesting():
    if "AlphaVantageBacktesting" not in _LUMIBOT_CACHE:
        from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
        _LUMIBOT_CACHE["AlphaVantageBacktesting"] = AlphaVantageFixedDataSource
    return _LUMIBOT_CACHE["AlphaVantageBacktesting"]


def _compute_accuracy(strat):
    from strategies.prediction_tracker import compute_prediction_accuracy
    return compute_prediction_accuracy(strat)


# ---------------------------------------------------------------------------
# Constants & Config
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join("regime_engine", "config.yaml")
REGIME_BINS = ["TREND_UP", "TREND_DN", "RANGE", "BREAKOUT_UP",
               "BREAKOUT_DN", "EXHAUST_REV", "LOWVOL", "HIGHVOL"]
TAXONOMY_LEVELS = ["kingdom", "phylum", "class_", "order", "family", "genus"]
GATE_FEATURES = ["default", "d_mass_dt", "d_lambda", "mass", "ofi_proxy", "dissipation_proxy"]
ML_MODELS = [
    ("Random Forest", "models/classifier_rf.pkl", "sklearn"),
    ("Gradient Boosting", "models/classifier_gb.pkl", "sklearn"),
    ("Logistic Regression", "models/classifier_lr.pkl", "sklearn"),
    ("LightGBM", "models/classifier_lgb.txt", "lgb"),
]

_TF_MINUTES = {"1D": 390, "4H": 240, "1H": 60, "15m": 15, "5m": 5, "1m": 1}
_TF_TIMESTEP = {"1D": "day", "4H": "minute", "1H": "minute",
                "15m": "minute", "5m": "minute", "1m": "minute"}


def _build_catalog() -> dict:
    return {
        "Buy & Hold": {
            "module": "strategies.buy_and_hold.BuyAndHold",
            "description": "Naive long-only baseline -- buy and hold forever.",
            "params": {"symbol": {"label": "Symbol", "type": "text", "default": "SPY"}},
        },
        "Momentum Rotation": {
            "module": "strategies.momentum.MomentumRotation",
            "description": "Rotates into strongest-momentum asset from a universe.",
            "params": {
                "symbols": {"label": "Universe", "type": "text", "default": "SPY,VEU,AGG,GLD"},
                "lookback_days": {"label": "Lookback (d)", "type": "number", "default": 20, "min": 5, "max": 200},
                "rebalance_days": {"label": "Rebalance (d)", "type": "number", "default": 5, "min": 1, "max": 60},
            },
        },
        "Mean Reversion": {
            "module": "strategies.mean_reversion.MeanReversion",
            "description": "Buy below SMA, sell above -- classic mean reversion.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "sma_period": {"label": "SMA Period", "type": "number", "default": 20, "min": 5, "max": 200},
            },
        },
        "RSI": {
            "module": "strategies.rsi.RSIStrategy",
            "description": "Relative Strength Index -- oversold/overbought signals.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "rsi_period": {"label": "Period", "type": "number", "default": 14, "min": 2, "max": 50},
                "oversold": {"label": "Oversold", "type": "number", "default": 30, "min": 10, "max": 50},
                "overbought": {"label": "Overbought", "type": "number", "default": 70, "min": 50, "max": 95},
            },
        },
        "Bollinger Bands": {
            "module": "strategies.bollinger.BollingerBands",
            "description": "Buy below lower band, sell above upper band.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "bb_period": {"label": "Period", "type": "number", "default": 20, "min": 5, "max": 100},
                "num_std": {"label": "Std Dev", "type": "number", "default": 2, "min": 1, "max": 4},
            },
        },
        "MACD": {
            "module": "strategies.macd.MACDStrategy",
            "description": "MACD crossover -- bullish/bearish momentum signal.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "fast_period": {"label": "Fast EMA", "type": "number", "default": 12, "min": 2, "max": 50},
                "slow_period": {"label": "Slow EMA", "type": "number", "default": 26, "min": 10, "max": 100},
                "signal_period": {"label": "Signal", "type": "number", "default": 9, "min": 2, "max": 30},
            },
        },
        "Dual SMA": {
            "module": "strategies.dual_sma.DualSMACrossover",
            "description": "Golden cross / death cross dual moving average.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "fast_period": {"label": "Fast SMA", "type": "number", "default": 10, "min": 2, "max": 100},
                "slow_period": {"label": "Slow SMA", "type": "number", "default": 50, "min": 10, "max": 300},
            },
        },
        "Channel Breakout": {
            "module": "strategies.breakout.ChannelBreakout",
            "description": "Donchian channel breakout/breakdown strategy.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "channel_period": {"label": "Period", "type": "number", "default": 20, "min": 5, "max": 100},
            },
        },
        "Wyckoff": {
            "module": "strategies.wyckoff.WyckoffStrategy",
            "description": "Wyckoff accumulation = BUY, distribution = SELL.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "lookback": {"label": "Lookback", "type": "number", "default": 30, "min": 10, "max": 120},
            },
        },
        "Liquidity Pools": {
            "module": "strategies.liquidity_pools.LiquidityPoolStrategy",
            "description": "Resting liquidity detection with sweep reversals.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "lookback": {"label": "Lookback", "type": "number", "default": 40, "min": 15, "max": 120},
                "swing_strength": {"label": "Swing", "type": "number", "default": 3, "min": 2, "max": 10},
            },
        },
        "ML Classifier (RF)": {
            "module": "strategies.ml_classifier_strategy.MLClassifierStrategy",
            "description": "Random Forest on full MFT feature set.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "min_confidence": {"label": "Min Conf", "type": "number", "default": 0.6, "min": 0.5, "max": 0.95},
            },
        },
        "ML Classifier (LightGBM)": {
            "module": "strategies.lightgbm_strategy.LightGBMStrategy",
            "description": "LightGBM gradient boosting on MFT features.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "min_confidence": {"label": "Min Conf", "type": "number", "default": 0.62, "min": 0.5, "max": 0.95},
            },
        },
        "Ensemble (RF+LGBM)": {
            "module": "strategies.ensemble_strategy.EnsembleMLStrategy",
            "description": "Both RF + LightGBM must agree for a signal.",
            "params": {"symbol": {"label": "Symbol", "type": "text", "default": "SPY"}},
        },
        "Phi-Bot (MFT)": {
            "module": "strategies.blended_mft_strategy.BlendedMFTStrategy",
            "description": "Full Market Field Theory -- default config weights.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "signal_threshold": {"label": "Threshold", "type": "number", "default": 0.15, "min": 0.05, "max": 0.50},
                "confidence_floor": {"label": "Conf Floor", "type": "number", "default": 0.30, "min": 0.10, "max": 0.80},
            },
        },
        "Plutus Bot (LLM)": {
            "module": "strategies.plutus_strategy.PlutusStrategy",
            "description": "0xroyce/Plutus LLM -- in-context learning from trade history.",
            "params": {
                "symbol": {"label": "Symbol", "type": "text", "default": "SPY"},
                "min_confidence": {"label": "Min Conf", "type": "number", "default": 0.60, "min": 0.50, "max": 0.95},
                "ollama_host": {"label": "Ollama Host", "type": "text", "default": "http://localhost:11434"},
            },
        },
    }


STRATEGY_CATALOG: dict = {}


def _ensure_catalog() -> dict:
    global STRATEGY_CATALOG
    if not STRATEGY_CATALOG:
        STRATEGY_CATALOG = _build_catalog()
    return STRATEGY_CATALOG


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_base_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# -- On-disk OHLCV cache ---------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _ohlcv_cache_path(symbol: str, start, end) -> str:
    s = str(start)[:10].replace("-", "")
    e = str(end)[:10].replace("-", "")
    return os.path.join(_CACHE_DIR, f"{symbol}_{s}_{e}.parquet")


def _cache_is_fresh(path: str, end) -> bool:
    if not os.path.exists(path):
        return False
    end_date = end if isinstance(end, date) else date.fromisoformat(str(end)[:10])
    if end_date >= date.today():
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        return age_hours < 24
    return True


def _load_ohlcv(symbol: str, start, end) -> Optional[pd.DataFrame]:
    """Load daily OHLCV — tries local cache → yfinance → Alpha Vantage."""
    cache_path = _ohlcv_cache_path(symbol, start, end)
    if _cache_is_fresh(cache_path, end):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Vendor fallback chain: yfinance (free, no key) → Alpha Vantage
    # ------------------------------------------------------------------
    start_s, end_s = str(start)[:10], str(end)[:10]

    # 1) yfinance — free, reliable for daily US equities
    try:
        import yfinance as yf
        tkr = yf.Ticker(symbol)
        raw = tkr.history(start=start_s, end=end_s, auto_adjust=True)
        if raw is not None and not raw.empty:
            # Normalize columns to lowercase
            raw.columns = [c.lower() for c in raw.columns]
            for col in ("open", "high", "low", "close", "volume"):
                if col not in raw.columns:
                    raise ValueError(f"yfinance missing column: {col}")
            raw = raw[["open", "high", "low", "close", "volume"]]
            raw.index = pd.to_datetime(raw.index)
            raw = raw.sort_index()
            try:
                raw.to_parquet(cache_path)
            except Exception:
                pass
            return raw
    except Exception as e:
        st.warning(f"yfinance failed for {symbol}: {e} — trying Alpha Vantage…")

    # 2) Alpha Vantage daily — falls through to AV REST API
    try:
        from phi.data.cache import fetch_and_cache
        raw = fetch_and_cache("alphavantage", symbol, "1D", start_s, end_s)
        if raw is not None and not raw.empty:
            try:
                raw.to_parquet(cache_path)
            except Exception:
                pass
            return raw
    except Exception as e:
        st.error(f"Download failed for {symbol}: {e}")

    return None


def _run_engine_with_config(ohlcv: pd.DataFrame, cfg: dict) -> Optional[dict]:
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


# ---------------------------------------------------------------------------
# Premium Sidebar
# ---------------------------------------------------------------------------
def build_sidebar(dev=None) -> dict:
    if dev is None:
        dev = get_device()

    # ── On mobile: render config in main area (not sidebar) ──
    if dev.is_mobile:
        return _build_config_mobile(dev)

    # ── Desktop: use standard sidebar ──
    return _build_config_sidebar(dev)


def _build_config_sidebar(dev) -> dict:
    """Desktop sidebar with brand, engine status, and config controls."""
    _premium_sidebar_brand()

    # Engine status indicator
    st.sidebar.markdown(f"""
    <div class="phi-engine-status">
        {_render_status_dot("live")}
        <span class="phi-engine-label">ENGINE ONLINE</span>
        <span class="phi-engine-time">{datetime.now().strftime("%H:%M:%S")}</span>
    </div>
    """, unsafe_allow_html=True)

    # Device badge
    dev_label = dev.device_type.value.upper()
    st.sidebar.markdown(f"""
    <div style="text-align:center;margin-bottom:0.8rem;">
        <span style="font-size:0.6rem;color:var(--phi-text-dim);letter-spacing:0.08em;
                     font-family:var(--font-mono);text-transform:uppercase;">
            MODE: {dev_label} &middot; {dev.screen_width}px
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(_render_section_header("", "BACKTEST CONFIG", ""), unsafe_allow_html=True)

    col_tf, col_n = st.sidebar.columns([1, 1])
    with col_tf:
        tf = st.selectbox("Timeframe", list(_TF_MINUTES.keys()), index=0, key="sb_timeframe")
    with col_n:
        n_bars = st.number_input("N Bars", value=2000, min_value=100, max_value=10_000, step=100, key="sb_nbars")

    bar_min = _TF_MINUTES[tf]
    trading_days_needed = (n_bars * bar_min) / 390
    calendar_days = int(trading_days_needed * 1.4) + 2
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=calendar_days)

    st.sidebar.markdown(f"""
    <div class="phi-info-bar">
        <div>
            <div class="phi-info-bar-label">Date Range</div>
            <div class="phi-info-bar-value">{start_dt.strftime('%Y-%m-%d')} &rarr; {end_dt.strftime('%Y-%m-%d')}</div>
            <div style="color:var(--phi-text-dim);font-size:0.7rem;margin-top:2px;">{n_bars:,} bars &middot; {calendar_days} cal days</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    b = st.sidebar.number_input("Budget ($)", value=100_000, min_value=1_000, step=10_000)
    bm = st.sidebar.text_input("Benchmark", value="SPY")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="phi-powered-by">
        <div class="phi-powered-label">Powered by</div>
        <div class="phi-powered-value">Lumibot + MFT Engine</div>
    </div>
    """, unsafe_allow_html=True)

    return _build_config_dict(tf, n_bars, b, bm)


def _build_config_mobile(dev) -> dict:
    """Mobile: render config in an expander in the main area instead of sidebar."""
    with st.expander("⚙️  Backtest Config", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            tf = st.selectbox("Timeframe", list(_TF_MINUTES.keys()), index=0, key="sb_timeframe")
            b = st.number_input("Budget ($)", value=100_000, min_value=1_000, step=10_000)
        with col2:
            n_bars = st.number_input("N Bars", value=2000, min_value=100, max_value=10_000, step=100, key="sb_nbars")
            bm = st.text_input("Benchmark", value="SPY")

        bar_min = _TF_MINUTES[tf]
        trading_days_needed = (n_bars * bar_min) / 390
        calendar_days = int(trading_days_needed * 1.4) + 2
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=calendar_days)

        st.markdown(f"""
        <div class="phi-info-bar">
            <div>
                <div class="phi-info-bar-label">Date Range</div>
                <div class="phi-info-bar-value">{start_dt.strftime('%Y-%m-%d')} &rarr; {end_dt.strftime('%Y-%m-%d')}</div>
                <div style="color:var(--phi-text-dim);font-size:0.7rem;margin-top:2px;">{n_bars:,} bars &middot; {calendar_days} cal days</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return _build_config_dict(tf, n_bars, b, bm)


def _build_config_dict(tf, n_bars, b, bm) -> dict:
    """Build the config dictionary from the control values."""
    bar_min = _TF_MINUTES[tf]
    trading_days_needed = (n_bars * bar_min) / 390
    calendar_days = int(trading_days_needed * 1.4) + 2
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=calendar_days)

    return {
        "start": datetime.combine(start_dt, datetime.min.time()),
        "end": datetime.combine(end_dt, datetime.min.time()),
        "budget": float(b),
        "benchmark": bm,
        "timeframe": tf,
        "n_bars": int(n_bars),
        "timestep": _TF_TIMESTEP[tf],
    }


def _run_backtest(strategy_class, params, config):
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")
    results, strat = strategy_class.run_backtest(
        datasource_class=_av_backtesting(),
        backtesting_start=config["start"], backtesting_end=config["end"],
        budget=config["budget"], benchmark_asset=config["benchmark"],
        parameters=params, api_key=av_api_key,
        show_plot=False, show_tearsheet=False, save_tearsheet=False,
        show_indicators=False, show_progress_bar=False, quiet_logs=True,
    )
    return results, strat


# ---------------------------------------------------------------------------
# Premium Result Displays
# ---------------------------------------------------------------------------
def _extract_scalar(val):
    if isinstance(val, dict):
        for key in ("drawdown", "value", "max_drawdown", "return"):
            if key in val:
                return val[key]
        for v in val.values():
            if isinstance(v, (int, float)):
                return v
        return None
    return val


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

        tr = _extract_scalar(_g(results, "total_return"))
        cagr = _extract_scalar(_g(results, "cagr"))
        dd = _extract_scalar(_g(results, "max_drawdown"))
        sh = _extract_scalar(_g(results, "sharpe"))

        kpis = [
            ("Total Return", f"{tr:+.1%}" if isinstance(tr, (int, float)) else "--",
             f"{tr:+.1%}" if isinstance(tr, (int, float)) and tr >= 0 else (f"{tr:.1%}" if isinstance(tr, (int, float)) else ""),
             "positive" if isinstance(tr, (int, float)) and tr >= 0 else "negative"),
            ("CAGR", f"{cagr:+.1%}" if isinstance(cagr, (int, float)) else "--", "", "neutral"),
            ("Max Drawdown", f"{dd:.1%}" if isinstance(dd, (int, float)) else "--", "", "negative"),
            ("Sharpe Ratio", f"{sh:.2f}" if isinstance(sh, (int, float)) else "--", "", "neutral"),
        ]
        st.markdown(_render_kpi_row(kpis), unsafe_allow_html=True)
    except Exception:
        pass

    # Premium equity curve
    try:
        pv = getattr(results, "portfolio_value", None) or (
            results.get("portfolio_value") if hasattr(results, "get") else None)
        if pv is not None and len(pv) > 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=pv, mode="lines",
                line=dict(color="#a855f7", width=2.5, shape="spline", smoothing=0.8),
                fill="tozeroy",
                fillcolor="rgba(168,85,247,0.06)",
                name="Portfolio Value",
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ))
            fig.add_hline(y=budget, line_dash="dot", line_color="rgba(148,163,184,0.2)",
                          annotation_text=f"Start: ${budget:,.0f}", annotation_font_color="#7a7a90")
            fig.update_layout(
                title=dict(text="EQUITY CURVE", font=dict(size=14, color="#7a7a90")),
                yaxis_title="Portfolio Value ($)", yaxis_tickformat="$,.0f",
            )
            _phi_chart(fig, height=380)
    except Exception:
        pass


def _show_accuracy(scorecard):
    if scorecard["total_predictions"] == 0:
        st.warning("No predictions recorded.")
        return

    kpis = [
        ("Accuracy", f"{scorecard['accuracy']:.1%}", "", "positive" if scorecard['accuracy'] > 0.5 else "negative"),
        ("Predictions", f"{scorecard['total_predictions']:,}", "", "neutral"),
        ("Correct", f"{scorecard['hits']:,}", "", "positive"),
        ("Edge", f"${scorecard['edge']:.4f}", "", "positive" if scorecard['edge'] > 0 else "negative"),
    ]
    st.markdown(_render_kpi_row(kpis), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("UP Accuracy", f"{scorecard['up_accuracy']:.1%}")
    col2.metric("DOWN Accuracy", f"{scorecard['down_accuracy']:.1%}")
    col3.metric("Wrong", f"{scorecard['misses']:,}")

    scored = scorecard["scored_log"]
    if len(scored) > 10:
        df = pd.DataFrame(scored)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["rolling"] = df["correct"].rolling(50, min_periods=10).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rolling"],
            mode="lines", line=dict(color="#f97316", width=2.5, shape="spline", smoothing=0.8),
            fill="tozeroy", fillcolor="rgba(249,115,22,0.05)",
            name="Rolling Accuracy",
            hovertemplate="%{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(148,163,184,0.2)",
                      annotation_text="50% baseline", annotation_font_color="#7a7a90")
        fig.update_layout(
            title=dict(text="50-BAR ROLLING ACCURACY", font=dict(size=14, color="#7a7a90")),
            yaxis_title="Accuracy", yaxis_tickformat=".0%",
        )
        _phi_chart(fig, height=300)

    with st.expander("Prediction Log", expanded=False):
        if scored:
            log = pd.DataFrame(scored)
            log["date"] = pd.to_datetime(log["date"]).dt.date
            log["actual_move"] = log["actual_move"].map(lambda x: f"${x:+.2f}")
            log["correct"] = log["correct"].map(lambda x: "HIT" if x else "MISS")
            st.dataframe(log[["date", "symbol", "signal", "price", "next_price", "actual_move", "correct"]],
                         use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 1: ML Model Status
# ---------------------------------------------------------------------------
def _check_sklearn(path):
    try:
        from regime_engine.ml_classifier import DirectionClassifier
        c = DirectionClassifier(); c.load(path)
        if not getattr(c, '_is_fitted', False): return False, None, None
        return True, getattr(c.model, "n_features_in_", "?"), list(getattr(c.model, "classes_", []))
    except Exception as e:
        return False, None, str(e)


def _check_lgb(path):
    try:
        from regime_engine.ml_classifier_lightgbm import LightGBMDirectionClassifier
        c = LightGBMDirectionClassifier(); c.load(path)
        if c.model is None: return False, None, None
        return True, c.model.num_feature(), c.model.num_trees()
    except Exception as e:
        return False, None, str(e)


def render_ml_status():
    dev = get_device()
    st.markdown(_render_section_header("", "ML MODEL STATUS", "AI CORE"), unsafe_allow_html=True)

    cols = st.columns(dev.cols_ml)
    any_missing = False
    for idx, (display, path, kind) in enumerate(ML_MODELS):
        with cols[idx % dev.cols_ml]:
            exists = os.path.exists(path)
            with st.container(border=True):
                if not exists:
                    any_missing = True
                    st.markdown(f"**{display}**")
                    st.markdown(f'{_render_status_dot("offline")} <span style="color:#ef4444;font-size:0.8rem;font-weight:600;">MISSING</span>', unsafe_allow_html=True)
                    st.caption(f"`{path}`")
                else:
                    ok, n_feat, extra = _check_sklearn(path) if kind == "sklearn" else _check_lgb(path)
                    kb = round(os.path.getsize(path) / 1024, 1)
                    if ok:
                        st.markdown(f"**{display}**")
                        st.markdown(f'{_render_status_dot("live")} <span style="color:#22c55e;font-size:0.8rem;font-weight:600;">READY</span>', unsafe_allow_html=True)
                        st.caption(f"{kb} KB")
                        if n_feat:
                            st.caption(f"Features: {n_feat}")
                    else:
                        any_missing = True
                        st.markdown(f"**{display}**")
                        st.markdown(f'{_render_status_dot("idle")} <span style="color:#f97316;font-size:0.8rem;font-weight:600;">ERROR</span>', unsafe_allow_html=True)

    st.markdown("---")

    csv = "historical_regime_features.csv"
    csv_ok = os.path.exists(csv)
    if csv_ok:
        try:
            n_rows = sum(1 for _ in open(csv, encoding="utf-8")) - 1
            n_cols = len(pd.read_csv(csv, nrows=1).columns)
            st.success(f"Training data ready: **{n_rows:,}** rows x **{n_cols}** cols -- Full MFT feature set")
        except Exception:
            st.warning("CSV exists but could not be read.")
    else:
        st.warning("No training CSV. Generate below or via the Fetch Data tab.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate MFT Training Data", disabled=csv_ok, use_container_width=True):
            with st.status("Generating MFT features...", expanded=True) as s:
                r = subprocess.run([sys.executable, "-X", "utf8", "generate_training_data.py"],
                                   capture_output=True, text=True, encoding="utf-8")
                st.code(r.stdout + r.stderr)
                s.update(label="Done" if r.returncode == 0 else "Failed",
                         state="complete" if r.returncode == 0 else "error")
            st.rerun()
    with c2:
        if st.button("Train All Models", disabled=not csv_ok, type="primary", use_container_width=True):
            with st.status("Training ML models...", expanded=True) as s:
                r = subprocess.run([sys.executable, "-X", "utf8", "train_ml_classifier.py"],
                                   capture_output=True, text=True, encoding="utf-8")
                st.code(r.stdout + r.stderr)
                s.update(label="Done" if r.returncode == 0 else "Failed",
                         state="complete" if r.returncode == 0 else "error")
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2: Fetch Data
# ---------------------------------------------------------------------------
def render_fetch_data():
    dev = get_device()
    st.markdown(_render_section_header("", "DATA ACQUISITION", "OHLCV"), unsafe_allow_html=True)

    if dev.is_phone:
        traw = st.text_input("Tickers (comma-separated)", value="SPY", key="fd_tickers")
        fs = st.date_input("From", value=date(2018, 1, 1), key="fd_start")
        fe = st.date_input("To", value=date(2024, 12, 31), key="fd_end")
    else:
        ca, cb, cc = st.columns([2, 1, 1])
        with ca: traw = st.text_input("Tickers (comma-separated)", value="SPY", key="fd_tickers")
        with cb: fs = st.date_input("From", value=date(2018, 1, 1), key="fd_start")
        with cc: fe = st.date_input("To", value=date(2024, 12, 31), key="fd_end")
    tickers = [t.strip().upper() for t in traw.split(",") if t.strip()]

    c1, c2 = st.columns(2)
    fetch_clicked = c1.button("Fetch Data", type="primary", use_container_width=True)
    gen_clicked = c2.button("Generate MFT Training CSV", use_container_width=True)

    if fetch_clicked and tickers:
        dfs = {}
        with st.spinner("Downloading OHLCV data..."):
            for sym in tickers:
                df = _load_ohlcv(sym, fs, fe)
                if df is not None:
                    dfs[sym] = df
        if dfs:
            st.session_state["fetched_data"] = dfs

    for sym, df in st.session_state.get("fetched_data", {}).items():
        st.markdown(f"#### {sym}")

        kpis_data = [("Rows", f"{len(df):,}", "", "neutral"),
                     ("From", str(df.index[0].date()), "", "neutral"),
                     ("To", str(df.index[-1].date()), "", "neutral")]
        if "close" in df.columns:
            ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
            kpis_data.append(("Return", f"{ret:+.1%}", f"{ret:+.1%}", "positive" if ret >= 0 else "negative"))
        st.markdown(_render_kpi_row(kpis_data), unsafe_allow_html=True)

        if "close" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["close"], mode="lines",
                line=dict(color="#a855f7", width=2, shape="spline", smoothing=0.6),
                fill="tozeroy", fillcolor="rgba(168,85,247,0.04)",
                name=f"{sym} Close",
                hovertemplate="$%{y:.2f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text=f"{sym} PRICE HISTORY", font=dict(size=13, color="#7a7a90")),
                yaxis_title="Price ($)", yaxis_tickformat="$,.2f",
            )
            _phi_chart(fig, height=300)

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
                out = engine.run(ohlcv)
                parts = [
                    out["features"],
                    out["logits"].rename(columns=lambda c: f"logit_{c}"),
                    out["regime_probs"].rename(columns=lambda c: f"prob_{c}"),
                    out["mix"][[c for c in out["mix"].columns
                                if c in ("composite_signal", "score", "c_field", "c_consensus", "c_liquidity")]],
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
                st.success("Training data saved. Go to **ML Model Status** to train models.")
            except Exception as e:
                s.update(label=f"Failed: {e}", state="error")
                st.exception(e)


# ---------------------------------------------------------------------------
# Tab 3: MFT Blender
# ---------------------------------------------------------------------------
def _blender_controls(base_cfg: dict, prefix: str) -> dict:
    overrides: dict = {}
    dev = get_device()

    st.markdown(_render_section_header("", "Taxonomy Smoothing", "EWM ALPHA"), unsafe_allow_html=True)
    st.caption("Alpha = 1 - persistence. **Low** = sticky. **High** = responsive.")
    n_cols = 2 if dev.is_phone else (3 if dev.is_tablet else len(TAXONOMY_LEVELS))
    cols = st.columns(n_cols)
    tax_smooth = {}
    for col_idx, level in enumerate(TAXONOMY_LEVELS):
        with cols[col_idx % n_cols]:
            default = base_cfg["taxonomy"]["smoothing"].get(level, 0.15)
            val = st.slider(level.replace("_", ""), 0.01, 0.60, float(default), 0.01, key=f"{prefix}_smooth_{level}")
            tax_smooth[level] = val
    overrides.setdefault("taxonomy", {})["smoothing"] = tax_smooth

    st.markdown(_render_section_header("", "Gate Steepness", "GAMMA"), unsafe_allow_html=True)
    st.caption("Controls tanh gate sharpness per feature. Higher = hard threshold.")
    gate_n_cols = 2 if dev.is_phone else (3 if dev.is_tablet else len(GATE_FEATURES))
    cols = st.columns(gate_n_cols)
    gate_steep = {}
    for col_idx, feat in enumerate(GATE_FEATURES):
        with cols[col_idx % gate_n_cols]:
            default = base_cfg["taxonomy"]["gate_steepness"].get(feat, 1.0)
            val = st.slider(feat, 0.1, 5.0, float(default), 0.1, key=f"{prefix}_gate_{feat}")
            gate_steep[feat] = val
    overrides["taxonomy"]["gate_steepness"] = gate_steep

    st.markdown(_render_section_header("", "MSL Kingdom Scale", ""), unsafe_allow_html=True)
    default_msl = float(base_cfg["taxonomy"].get("msl_kingdom_scale", 0.8))
    overrides["taxonomy"]["msl_kingdom_scale"] = st.slider(
        "MSL Scale", 0.0, 2.0, default_msl, 0.05, key=f"{prefix}_msl_scale")

    st.markdown(_render_section_header("", "Mixer / Confidence", "INTERFACE 4,6"), unsafe_allow_html=True)
    conf_cfg = base_cfg["confidence"]
    if dev.is_phone:
        d1, d2 = st.columns(2)
        d3, d4 = st.columns(2)
    else:
        d1, d2, d3, d4 = st.columns(4)
    conf_overrides: dict = {}
    with d1:
        conf_overrides["affinity_blend"] = st.slider("Affinity", 0.0, 1.0,
            float(conf_cfg.get("affinity_blend", 0.5)), 0.05, key=f"{prefix}_affinity")
    with d2:
        conf_overrides["interaction_alpha"] = st.slider("Interaction", 0.0, 1.0,
            float(conf_cfg.get("interaction_alpha", 0.7)), 0.05, key=f"{prefix}_interaction")
    with d3:
        conf_overrides["liquidity_volume_scale"] = st.slider("Vol Scale", 0.1, 2.0,
            float(conf_cfg.get("liquidity_volume_scale", 0.5)), 0.05, key=f"{prefix}_vol_scale")
    with d4:
        conf_overrides["liquidity_gap_scale"] = st.slider("Gap Scale", 0.1, 2.0,
            float(conf_cfg.get("liquidity_gap_scale", 0.5)), 0.05, key=f"{prefix}_gap_scale")
    overrides["confidence"] = conf_overrides

    st.markdown(_render_section_header("", "Projection AR(1)", "PER REGIME"), unsafe_allow_html=True)
    proj_cfg = base_cfg["projection"]["regimes"]
    proj_overrides: dict = {"regimes": {}}
    for regime in REGIME_BINS:
        rp = proj_cfg.get(regime, {})
        with st.expander(f"{regime}  (mu={rp.get('mu',0):+.2f}, phi={rp.get('phi',0):.2f})"):
            r1, r2, r3, r4 = st.columns(4)
            proj_overrides["regimes"][regime] = {
                "mu": r1.slider("mu", -1.0, 1.0, float(rp.get("mu", 0.0)), 0.05, key=f"{prefix}_proj_{regime}_mu"),
                "phi": r2.slider("phi", -1.0, 1.0, float(rp.get("phi", 0.3)), 0.05, key=f"{prefix}_proj_{regime}_phi"),
                "beta": r3.slider("beta", -1.0, 1.0, float(rp.get("beta", 0.0)), 0.05, key=f"{prefix}_proj_{regime}_beta"),
                "sigma": r4.slider("sigma", 0.01, 1.0, float(rp.get("sigma", 0.2)), 0.01, key=f"{prefix}_proj_{regime}_sigma"),
            }
    overrides["projection"] = proj_overrides
    return overrides


def _run_and_display_pipeline(ohlcv: pd.DataFrame, cfg: dict, sym: str):
    with st.spinner("Running full MFT pipeline..."):
        out = _run_engine_with_config(ohlcv, cfg)
    if out is None:
        return

    feat_df = out["features"]
    logits_df = out["logits"]
    regime_df = out["regime_probs"]
    mix_df = out["mix"]
    signals_df = out["signals"]
    weights_df = out["weights"]
    proj_exp = out["projections"]["expected"]
    proj_var = out["projections"]["variance"]

    st.session_state["blender_out"] = out
    st.session_state["blender_ohlcv"] = ohlcv
    st.session_state["blender_cfg"] = cfg

    # Stage 1: Features
    with st.expander("STAGE 1 -- Feature Engine", expanded=False):
        feat_cols = list(feat_df.columns)
        sel_feats = st.multiselect("Features", feat_cols, default=feat_cols[:6], key="blender_feat_sel")
        if sel_feats:
            fig = go.Figure()
            for i, col in enumerate(sel_feats):
                fig.add_trace(go.Scatter(
                    x=feat_df.index[-252:], y=feat_df[col].tail(252),
                    mode="lines", name=col,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
                ))
            fig.update_layout(title=dict(text="FEATURE ENGINE OUTPUT", font=dict(size=13, color="#7a7a90")))
            _phi_chart(fig, height=350)
        st.caption(f"{len(feat_cols)} computed features")

    # Stage 2: Taxonomy Logits
    with st.expander("STAGE 2 -- Taxonomy Logits", expanded=True):
        node_groups = {
            "Kingdom": [c for c in logits_df.columns if c in ("DIR", "NDR", "TRN")],
            "Phylum": [c for c in logits_df.columns if c in ("LV", "NV", "HV")],
            "Class": [c for c in logits_df.columns if c in ("PT", "PX", "TE", "BR", "RR", "AR", "SR", "RB", "FB")],
        }
        for grp_name, grp_cols in node_groups.items():
            if grp_cols:
                fig = go.Figure()
                for i, col in enumerate(grp_cols):
                    fig.add_trace(go.Scatter(
                        x=logits_df.index[-252:], y=logits_df[col].tail(252),
                        mode="lines", name=col,
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.5),
                    ))
                fig.update_layout(title=dict(text=f"TAXONOMY: {grp_name.upper()}", font=dict(size=13, color="#7a7a90")))
                _phi_chart(fig, height=280)

    # Stage 3: Regime Probabilities
    with st.expander("STAGE 3 -- Probability Field", expanded=True):
        tail = regime_df.tail(252)
        fig = go.Figure()
        for regime_name in tail.columns:
            color = REGIME_COLORS.get(regime_name, "#94a3b8")
            fig.add_trace(go.Scatter(
                x=tail.index, y=tail[regime_name],
                mode="lines", name=regime_name, stackgroup="one",
                line=dict(width=0.5, color=color),
                fillcolor=color.replace(")", ",0.4)").replace("rgb", "rgba") if "rgb" in color else color + "66",
            ))
        fig.update_layout(
            title=dict(text="8-REGIME PROBABILITY FIELD", font=dict(size=14, color="#7a7a90")),
            yaxis_title="Probability", yaxis_tickformat=".0%",
        )
        _phi_chart(fig, height=400)

        # Latest regime breakdown
        latest = regime_df.iloc[-1].sort_values(ascending=False)
        regime_cols = st.columns(4)
        for i, (r, p) in enumerate(latest.items()):
            color = REGIME_COLORS.get(r, "#94a3b8")
            regime_cols[i % 4].markdown(f"""
            <div style="text-align:center;padding:10px;background:rgba(18,18,26,0.7);border-radius:10px;
                        border:1px solid {color}22;margin-bottom:4px;transition:all 0.3s ease;">
                <div style="color:#7a7a90;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;">{r}</div>
                <div style="color:{color};font-size:1.3rem;font-weight:700;font-family:'JetBrains Mono',monospace;">{p:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    # Stage 4: Signals + Weights
    with st.expander("STAGE 4 -- Signals & Weights", expanded=False):
        sa, sb = st.tabs(["Signals", "Validity Weights"])
        with sa:
            fig = go.Figure()
            for i, col in enumerate(signals_df.columns[:8]):
                fig.add_trace(go.Scatter(
                    x=signals_df.index[-252:], y=signals_df[col].tail(252),
                    mode="lines", name=col,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.2),
                ))
            fig.update_layout(title=dict(text="INDICATOR SIGNALS", font=dict(size=13, color="#7a7a90")))
            _phi_chart(fig, height=320)
        with sb:
            fig = go.Figure()
            for i, col in enumerate(weights_df.columns[:8]):
                fig.add_trace(go.Scatter(
                    x=weights_df.index[-252:], y=weights_df[col].tail(252),
                    mode="lines", name=col,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.2),
                ))
            fig.update_layout(title=dict(text="VALIDITY WEIGHTS", font=dict(size=13, color="#7a7a90")))
            _phi_chart(fig, height=320)

    # Stage 5: Mixer
    with st.expander("STAGE 5 -- Mixer / Composite", expanded=True):
        conf_cols = [c for c in mix_df.columns
                     if c in ("composite_signal", "score", "c_field", "c_consensus", "c_liquidity")]
        if conf_cols:
            fig = make_subplots(rows=1, cols=1)
            colors_map = {"composite_signal": "#a855f7", "score": "#f97316",
                          "c_field": "#22c55e", "c_consensus": "#06b6d4", "c_liquidity": "#eab308"}
            for col in conf_cols:
                fig.add_trace(go.Scatter(
                    x=mix_df.index[-252:], y=mix_df[col].tail(252),
                    mode="lines", name=col,
                    line=dict(color=colors_map.get(col, "#94a3b8"), width=2),
                ))
            fig.update_layout(title=dict(text="COMPOSITE SCORE & CONFIDENCE", font=dict(size=14, color="#7a7a90")))
            _phi_chart(fig, height=350)

        last_mix = mix_df.iloc[-1]
        kpis = [
            ("Composite", f"{last_mix.get('composite_signal', 0):+.3f}", "", "neutral"),
            ("Score", f"{last_mix.get('score', 0):+.3f}", "", "neutral"),
            ("C_field", f"{last_mix.get('c_field', 0):.3f}", "", "neutral"),
            ("C_consensus", f"{last_mix.get('c_consensus', 0):.3f}", "", "neutral"),
        ]
        st.markdown(_render_kpi_row(kpis), unsafe_allow_html=True)

        overall = (last_mix.get("c_field", 0) * last_mix.get("c_consensus", 0) * last_mix.get("c_liquidity", 0))
        if overall >= 0.1:
            st.success(f"Overall confidence: **{overall:.3f}** -- Signal is tradeable")
        else:
            st.warning(f"Overall confidence: **{overall:.3f}** -- Below trade threshold")

    # Stage 6: Projections
    with st.expander("STAGE 6 -- Projections", expanded=False):
        pe, pv = st.tabs(["Expected Value", "Variance"])
        with pe:
            fig = go.Figure()
            for i, col in enumerate(proj_exp.columns[:6]):
                fig.add_trace(go.Scatter(
                    x=proj_exp.index[-252:], y=proj_exp[col].tail(252),
                    mode="lines", name=col,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.2),
                ))
            fig.update_layout(title=dict(text="AR(1) EXPECTED VALUES", font=dict(size=13, color="#7a7a90")))
            _phi_chart(fig, height=320)
        with pv:
            fig = go.Figure()
            for i, col in enumerate(proj_var.columns[:6]):
                fig.add_trace(go.Scatter(
                    x=proj_var.index[-252:], y=proj_var[col].tail(252),
                    mode="lines", name=col,
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.2),
                ))
            fig.update_layout(title=dict(text="MIXTURE VARIANCE", font=dict(size=13, color="#7a7a90")))
            _phi_chart(fig, height=320)


def render_mft_blender(config: dict):
    dev = get_device()
    st.markdown(_render_section_header("", "MFT BLENDER", "REAL-TIME"), unsafe_allow_html=True)
    st.caption("All sliders map directly to `config.yaml`. Changes propagate through the full MFT pipeline.")

    base_cfg = _load_base_config()

    if dev.is_phone:
        sym = st.text_input("Symbol", value="SPY", key="blender_sym")
        bl_start = st.date_input("From", value=date(2020, 1, 1), key="blender_start")
        bl_end = st.date_input("To", value=date(2024, 12, 31), key="blender_end")
        auto_refresh = st.toggle("Auto-refresh", value=False, key="blender_auto")
        refresh_secs = st.number_input("Interval (s)", value=30, min_value=5, max_value=300,
                                        key="blender_interval") if auto_refresh else None
    else:
        col_sym, col_s, col_e, col_auto = st.columns([2, 1, 1, 1])
        with col_sym: sym = st.text_input("Symbol", value="SPY", key="blender_sym")
        with col_s: bl_start = st.date_input("From", value=date(2020, 1, 1), key="blender_start")
        with col_e: bl_end = st.date_input("To", value=date(2024, 12, 31), key="blender_end")
        with col_auto:
            auto_refresh = st.toggle("Auto-refresh", value=False, key="blender_auto")
            refresh_secs = st.number_input("Interval (s)", value=30, min_value=5, max_value=300,
                                            key="blender_interval") if auto_refresh else None

    with st.form("blender_form"):
        overrides = _blender_controls(base_cfg, "bl")
        submitted = st.form_submit_button("Run Pipeline", type="primary", use_container_width=True)

    if submitted or (auto_refresh and st.session_state.get("blender_auto_ran")):
        ohlcv = _load_ohlcv(sym, bl_start, bl_end)
        if ohlcv is None:
            return
        merged_cfg = _deep_merge(base_cfg, overrides)
        _run_and_display_pipeline(ohlcv, merged_cfg, sym)
        st.session_state["blender_auto_ran"] = True

    if auto_refresh and refresh_secs and st.session_state.get("blender_auto_ran"):
        st.caption(f"Auto-refresh every {refresh_secs}s -- last: {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(int(refresh_secs))
        st.rerun()
    elif not auto_refresh and not st.session_state.get("blender_auto_ran"):
        st.info("Adjust parameters and click **Run Pipeline** to see the full MFT pipeline output.")

    # Cached results display
    if not submitted and st.session_state.get("blender_out") is not None:
        st.caption("Showing last run results:")
        out = st.session_state["blender_out"]
        with st.expander("STAGE 3 -- Regime Probabilities", expanded=True):
            tail = out["regime_probs"].tail(252)
            fig = go.Figure()
            for regime_name in tail.columns:
                color = REGIME_COLORS.get(regime_name, "#94a3b8")
                fig.add_trace(go.Scatter(
                    x=tail.index, y=tail[regime_name],
                    mode="lines", name=regime_name, stackgroup="one",
                    line=dict(width=0.5, color=color),
                ))
            fig.update_layout(title=dict(text="REGIME PROBABILITIES", font=dict(size=13, color="#7a7a90")))
            _phi_chart(fig, height=350)

    # Blended backtest
    st.markdown("---")
    st.markdown(_render_section_header("", "BLENDED BACKTEST", ""), unsafe_allow_html=True)
    if st.button("Run Blended Backtest", use_container_width=True):
        if st.session_state.get("blender_out") is None:
            st.warning("Run the pipeline first.")
        else:
            merged_cfg = _deep_merge(base_cfg, overrides if submitted else {})
            with st.status("Running backtest...", expanded=True) as s:
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        _BlendedMFT = _strat("strategies.blended_mft_strategy.BlendedMFTStrategy")
                        results, strat = _run_backtest(_BlendedMFT, {"symbol": sym, "indicator_weights": {}}, config)
                    sc = _compute_accuracy(strat)
                    s.update(label=f"Complete -- {sc['accuracy']:.1%}", state="complete")
                    bt1, bt2 = st.tabs(["Portfolio", "Accuracy"])
                    with bt1: _show_portfolio(results, config["budget"])
                    with bt2: _show_accuracy(sc)
                except Exception as e:
                    s.update(label=f"Failed: {e}", state="error")
                    st.exception(e)


# ---------------------------------------------------------------------------
# Tab 4: Phi-Bot
# ---------------------------------------------------------------------------
def render_phi_bot(config: dict):
    dev = get_device()
    st.markdown(_render_section_header("", "PHI-BOT", "FULL MFT SYSTEM"), unsafe_allow_html=True)
    st.caption("Pure Market Field Theory -- no manual tuning. The full pipeline drives every signal.")

    scan_tab, bt_tab = st.tabs(["Regime Scanner", "Phi-Bot Backtest"])

    with scan_tab:
        st.markdown(_render_section_header("", "Universe Scanner", "MULTI-ASSET"), unsafe_allow_html=True)
        if dev.is_phone:
            universe_raw = st.text_input("Universe", value="SPY, QQQ, AAPL, NVDA", key="phi_universe")
            scan_start = st.date_input("From", value=date(2022, 1, 1), key="phi_scan_start")
            scan_end = st.date_input("To", value=date(2024, 12, 31), key="phi_scan_end")
        else:
            ca, cb, cc = st.columns([3, 1, 1])
            with ca: universe_raw = st.text_input("Universe", value="SPY, QQQ, AAPL, NVDA, TSLA, MSFT, AMZN, GLD", key="phi_universe")
            with cb: scan_start = st.date_input("From", value=date(2022, 1, 1), key="phi_scan_start")
            with cc: scan_end = st.date_input("To", value=date(2024, 12, 31), key="phi_scan_end")

        if dev.is_phone:
            sort_col = st.selectbox("Sort by", ["score", "composite_signal", "c_field", "c_consensus"], key="phi_sort")
            live_scan = st.toggle("Live scan", key="phi_live")
        else:
            col_sort, col_auto = st.columns([2, 1])
            sort_col = col_sort.selectbox("Sort by", ["score", "composite_signal", "c_field", "c_consensus"], key="phi_sort")
            live_scan = col_auto.toggle("Live scan", key="phi_live")
        if live_scan:
            refresh_s = st.slider("Refresh interval (s)", 15, 300, 60, key="phi_refresh_s")

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
            display_cols = [c for c in ["ticker", "score", "composite_signal", "c_field", "c_consensus", "c_liquidity", "top_species", "top_species_desc"]
                           if c in scan_df.columns]
            st.dataframe(scan_df[display_cols], use_container_width=True, hide_index=True)

            # Regime prob heatmap
            prob_cols = [c for c in scan_df.columns if c.startswith("p_")]
            if prob_cols:
                prob_data = scan_df.set_index("ticker")[prob_cols].rename(columns=lambda c: c.replace("p_", ""))
                fig = go.Figure(data=go.Heatmap(
                    z=prob_data.values, x=prob_data.columns.tolist(), y=prob_data.index.tolist(),
                    colorscale=[[0, "#06060a"], [0.3, "#7c3aed"], [0.7, "#a855f7"], [1, "#f97316"]],
                    hovertemplate="%{y} - %{x}: %{z:.1%}<extra></extra>",
                ))
                fig.update_layout(
                    title=dict(text="REGIME PROBABILITY HEATMAP", font=dict(size=14, color="#7a7a90")),
                    xaxis_title="Regime", yaxis_title="Ticker",
                )
                _phi_chart(fig, height=max(200, len(prob_data) * (25 if dev.is_phone else 35)))

        if live_scan and st.session_state.get("scan_results") is not None:
            st.caption(f"Next refresh in {refresh_s}s...")
            time.sleep(refresh_s)
            st.rerun()

    with bt_tab:
        st.markdown(_render_section_header("", "Phi-Bot Backtest", ""), unsafe_allow_html=True)
        bt_sym = st.text_input("Symbol", value="SPY", key="phi_bt_sym")
        ct, cc2 = st.columns(2)
        threshold = ct.slider("Signal threshold", 0.05, 0.50, 0.15, 0.05, key="phi_thresh")
        conf_floor = cc2.slider("Confidence floor", 0.10, 0.80, 0.30, 0.05, key="phi_conf")

        if st.button("Run Phi-Bot Backtest", type="primary", use_container_width=True):
            with st.status("Running Phi-Bot...", expanded=True) as s:
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        _BlendedMFT = _strat("strategies.blended_mft_strategy.BlendedMFTStrategy")
                        results, strat = _run_backtest(_BlendedMFT,
                            {"symbol": bt_sym, "indicator_weights": {},
                             "signal_threshold": threshold, "confidence_floor": conf_floor}, config)
                    sc = _compute_accuracy(strat)
                    s.update(label=f"Phi-Bot -- {sc['accuracy']:.1%}", state="complete")
                    t1, t2 = st.tabs(["Portfolio", "Accuracy"])
                    with t1: _show_portfolio(results, config["budget"])
                    with t2: _show_accuracy(sc)
                except Exception as e:
                    s.update(label=f"Failed: {e}", state="error")
                    st.exception(e)


# ---------------------------------------------------------------------------
# Tab 5: Classic Backtests
# ---------------------------------------------------------------------------
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
    for k in ("lookback_days", "rebalance_days", "sma_period", "rsi_period", "oversold",
              "overbought", "bb_period", "fast_period", "slow_period", "signal_period",
              "channel_period", "lookback", "swing_strength"):
        if k in resolved:
            resolved[k] = int(resolved[k])
    return resolved


def render_backtests(config: dict):
    dev = get_device()
    st.markdown(_render_section_header("", "STRATEGY ARENA", "MULTI-STRATEGY"), unsafe_allow_html=True)
    st.caption("Select strategies, configure parameters, and run head-to-head backtests.")

    selected = {}
    items = list(_ensure_catalog().items())
    cards_per_row = dev.cols_strategy
    for row_start in range(0, len(items), cards_per_row):
        cols = st.columns(cards_per_row)
        for col, (name, info) in zip(cols, items[row_start:row_start + cards_per_row]):
            with col:
                with st.container(border=True):
                    enabled, params = _strategy_card(name, info)
                    if enabled and params is not None:
                        selected[name] = {"cls": _strat(info["module"]), "params": _resolve(name, params)}

    st.markdown("---")
    if not selected:
        st.info("Enable at least one strategy to begin.")
        return

    st.markdown(f"""
    <div class="phi-info-bar">
        <span class="phi-info-bar-label">Ready:</span>
        <span class="phi-info-bar-value">{', '.join(selected.keys())}</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Backtests", type="primary", use_container_width=True):
        all_sc, all_res = {}, {}
        for name, entry in selected.items():
            with st.status(f"Running {name}...", expanded=True) as s:
                try:
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        results, strat = _run_backtest(entry["cls"], entry["params"], config)
                    sc = _compute_accuracy(strat)
                    all_sc[name] = sc
                    all_res[name] = results
                    s.update(label=f"{name} -- {sc['accuracy']:.1%}", state="complete")
                except Exception as e:
                    s.update(label=f"{name} -- failed", state="error")
                    st.error(str(e))

        for name, sc in all_sc.items():
            st.markdown(f"### {name}")
            t1, t2 = st.tabs(["Portfolio", "Accuracy"])
            with t1: _show_portfolio(all_res.get(name), config["budget"])
            with t2: _show_accuracy(sc)

        # Comparison table
        if len(all_sc) > 1:
            st.markdown("---")
            st.markdown(_render_section_header("", "HEAD-TO-HEAD COMPARISON", ""), unsafe_allow_html=True)
            rows = [{"Strategy": n, "Accuracy": f"{sc['accuracy']:.1%}",
                     "Predictions": sc["total_predictions"],
                     "Edge": f"${sc['edge']:.4f}"}
                    for n, sc in all_sc.items() if sc["total_predictions"] > 0]
            rows.sort(key=lambda r: float(r["Accuracy"].strip("%")) / 100, reverse=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 6: Plutus Bot
# ---------------------------------------------------------------------------
def _get_plutus_advisor(host: str, model: str, min_conf: float):
    cache_key = f"_plutus_advisor_{host}_{model}"
    if cache_key not in st.session_state:
        from regime_engine.plutus_advisor import PlutusAdvisor
        st.session_state[cache_key] = PlutusAdvisor(model=model, host=host, min_conf=min_conf)
    return st.session_state[cache_key]


def render_plutus_bot(config: dict):
    dev = get_device()
    st.markdown(_render_section_header("", "PLUTUS BOT", "LLM POWERED"), unsafe_allow_html=True)
    st.caption("Powered by 0xroyce/plutus -- LLaMA 3.1-8B fine-tuned on 394 finance & trading books.")

    with st.expander("Ollama Connection", expanded=True):
        if dev.is_phone:
            ollama_host = st.text_input("Ollama Host", value="http://localhost:11434", key="plutus_host")
            plutus_model = st.text_input("Model", value="0xroyce/plutus", key="plutus_model_name")
            check_btn = st.button("Check Connection", use_container_width=True)
        else:
            c1, c2, c3 = st.columns([3, 3, 1])
            with c1: ollama_host = st.text_input("Ollama Host", value="http://localhost:11434", key="plutus_host")
            with c2: plutus_model = st.text_input("Model", value="0xroyce/plutus", key="plutus_model_name")
            with c3:
                st.write(""); st.write("")
                check_btn = st.button("Check", use_container_width=True)

        if check_btn:
            with st.spinner("Checking..."):
                try:
                    from regime_engine.plutus_advisor import PlutusAdvisor
                    adv = PlutusAdvisor(model=plutus_model, host=ollama_host)
                    if adv.is_available():
                        st.success(f"Model `{plutus_model}` is ready.")
                    else:
                        st.warning(f"Model not found. Run: `ollama pull {plutus_model}`")
                except Exception as e:
                    st.error(f"Cannot reach Ollama: {e}")

    st.markdown("---")
    ask_tab, bt_tab, journal_tab = st.tabs(["Ask Plutus", "Backtest", "Trade Journal"])

    with ask_tab:
        st.markdown(_render_section_header("", "Ask Plutus", "RECOMMENDATION"), unsafe_allow_html=True)
        if dev.is_phone:
            ask_sym = st.text_input("Symbol", value="SPY", key="plutus_ask_sym")
            ask_start = st.date_input("From", value=date(2023, 1, 1), key="plutus_ask_start")
            ask_end = st.date_input("To", value=date(2024, 12, 31), key="plutus_ask_end")
        else:
            qa, qb, qc = st.columns([2, 1, 1])
            with qa: ask_sym = st.text_input("Symbol", value="SPY", key="plutus_ask_sym")
            with qb: ask_start = st.date_input("From", value=date(2023, 1, 1), key="plutus_ask_start")
            with qc: ask_end = st.date_input("To", value=date(2024, 12, 31), key="plutus_ask_end")
        ask_conf = st.slider("Minimum confidence", 0.50, 0.95, 0.60, 0.05, key="plutus_ask_conf")

        if st.button("Get Recommendation", type="primary", use_container_width=True):
            with st.status("Consulting Plutus...", expanded=True) as s:
                try:
                    ohlcv = _load_ohlcv(ask_sym, ask_start, ask_end)
                    if ohlcv is None or ohlcv.empty:
                        s.update(label="No data", state="error")
                    else:
                        mft_out = None
                        try:
                            from regime_engine.scanner import RegimeEngine
                            mft_out = RegimeEngine(_load_base_config()).run(ohlcv)
                        except Exception:
                            pass
                        from regime_engine.plutus_advisor import PlutusAdvisor, build_market_brief
                        advisor = _get_plutus_advisor(ollama_host, plutus_model, ask_conf)
                        if not advisor.is_available():
                            s.update(label="Plutus offline", state="error")
                        else:
                            ohlcv_sum, mft_sig = build_market_brief(ohlcv, mft_out)
                            price = float(ohlcv["close"].iloc[-1])
                            decision = advisor.recommend(ask_sym, ohlcv_sum, mft_sig, price)
                            st.session_state["plutus_last_decision"] = decision
                            s.update(label=f"Plutus says: {decision.signal} ({decision.confidence:.0%})", state="complete")
                except Exception as e:
                    s.update(label=f"Error: {e}", state="error")
                    st.exception(e)

        dec = st.session_state.get("plutus_last_decision")
        if dec is not None:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem;background:rgba(18,18,26,0.7);
                        border:1px solid rgba(168,85,247,0.12);border-radius:16px;margin:1.5rem 0;
                        backdrop-filter:blur(16px);">
                {_render_signal_badge(dec.signal)}
                <div style="color:#c084fc;font-size:1.5rem;font-weight:700;margin-top:0.8rem;
                            font-family:'JetBrains Mono',monospace;">
                    {dec.confidence:.0%} CONFIDENCE
                </div>
                <div style="color:#7a7a90;font-size:0.85rem;margin-top:0.5rem;font-weight:600;">
                    {"ACTIONABLE" if dec.is_actionable(ask_conf) else "BELOW THRESHOLD"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Reasoning:** {dec.reasoning}")
            st.markdown(f"**Risk:** {dec.risk_note}")
            if dec.raw:
                with st.expander("Raw LLM Output"):
                    st.code(dec.raw, language="json")

    with bt_tab:
        st.markdown(_render_section_header("", "Plutus Backtest", ""), unsafe_allow_html=True)
        bt_sym = st.text_input("Symbol", value="SPY", key="plutus_bt_sym")
        bt_c1, bt_c2 = st.columns(2)
        bt_conf = bt_c1.slider("Min confidence", 0.50, 0.95, 0.60, 0.05, key="plutus_bt_conf")
        bt_pos = bt_c2.slider("Position size %", 0.50, 1.00, 0.95, 0.05, key="plutus_bt_pos")

        if st.button("Run Plutus Backtest", type="primary", use_container_width=True):
            with st.status("Running Plutus Bot...", expanded=True) as s:
                try:
                    _PlutusStrat = _strat("strategies.plutus_strategy.PlutusStrategy")
                    buf = io.StringIO()
                    with redirect_stdout(buf), redirect_stderr(buf):
                        results, strat = _run_backtest(_PlutusStrat,
                            {"symbol": bt_sym, "min_confidence": bt_conf,
                             "position_pct": bt_pos, "ollama_host": ollama_host,
                             "plutus_model": plutus_model}, config)
                    sc = _compute_accuracy(strat)
                    s.update(label=f"Plutus -- {sc['accuracy']:.1%}", state="complete")
                    p1, p2 = st.tabs(["Portfolio", "Accuracy"])
                    with p1: _show_portfolio(results, config["budget"])
                    with p2: _show_accuracy(sc)
                except Exception as e:
                    s.update(label=f"Failed: {e}", state="error")
                    st.exception(e)

    with journal_tab:
        st.markdown(_render_section_header("", "Trade Journal", "IN-CONTEXT LEARNING"), unsafe_allow_html=True)
        adv = st.session_state.get(f"_plutus_advisor_{ollama_host}_{plutus_model}")
        if adv is None:
            st.info("No advisor session yet. Use 'Ask Plutus' or run a backtest first.")
        else:
            journal = adv.get_journal()
            if not journal:
                st.info("Journal empty -- no decisions recorded yet.")
            else:
                hit = adv.journal_accuracy()
                kpis = [
                    ("Decisions", str(len(journal)), "", "neutral"),
                    ("Settled", str(sum(1 for e in journal if e.get("pnl_pct") is not None)), "", "neutral"),
                    ("Hit Rate", f"{hit:.1%}" if hit else "--", "", "positive" if hit and hit > 0.5 else "negative"),
                ]
                st.markdown(_render_kpi_row(kpis), unsafe_allow_html=True)
                df_j = pd.DataFrame(journal)
                for col in ("entry_price", "exit_price"):
                    if col in df_j:
                        df_j[col] = df_j[col].apply(lambda x: f"{x:.2f}" if x is not None else "open")
                if "pnl_pct" in df_j:
                    df_j["pnl_pct"] = df_j["pnl_pct"].apply(lambda x: f"{x:+.2f}%" if x is not None else "--")
                st.dataframe(df_j, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 7: System Status
# ---------------------------------------------------------------------------
def render_system_status():
    dev = get_device()
    st.markdown(_render_section_header("", "SYSTEM STATUS", "ENGINE HEALTH"), unsafe_allow_html=True)

    if st.button("Run Health Check", type="primary", use_container_width=True):
        with st.status("Checking engine health...", expanded=True) as s:
            try:
                health = run_engine_health_check()
                st.session_state["engine_health"] = health
                s.update(label="Complete" if not health.get("error") else "Issues detected",
                         state="complete" if not health.get("error") else "error")
            except Exception as e:
                st.session_state["engine_health"] = {"ok": False, "error": str(e), "components": {}, "optional": {}}
                s.update(label="Error", state="error")

    health = st.session_state.get("engine_health")
    if health is None:
        return

    if health.get("error") and not health.get("components"):
        st.error(health["error"])
        return

    if health.get("ok"):
        st.success("All MFT pipeline components connected and operational.")
    else:
        st.warning("One or more components have issues.")

    n_status_cols = 1 if dev.is_phone else (2 if dev.is_tablet else 3)
    cols = st.columns(n_status_cols)
    for idx, (name, c) in enumerate(health.get("components", {}).items()):
        with cols[idx % n_status_cols]:
            with st.container(border=True):
                status = "live" if c.get("ok") else "offline"
                st.markdown(f'{_render_status_dot(status)} **{name}**', unsafe_allow_html=True)
                st.caption(c.get("message", ""))

    with st.expander("Optional Components"):
        for k, v in health.get("optional", {}).items():
            st.text(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Phi-nance | Premium Quant Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_premium_css()

    # Device detection -- JS already injected by _inject_premium_css
    dev = detect_device(skip_js=True)

    # On desktop, re-expand the sidebar now that we know the device
    if dev.is_desktop:
        st.set_page_config(initial_sidebar_state="expanded")

    # Mobile header (phones/tablets get a sticky header)
    _render_mobile_header()

    config = build_sidebar(dev)

    # Hero Section
    _render_hero()

    # Feature Showcase -- auto-collapse on mobile
    with st.expander("Platform Capabilities", expanded=dev.is_desktop):
        _render_features()

    st.markdown('<div class="phi-gradient-bar"></div>', unsafe_allow_html=True)

    # Main navigation tabs
    tabs = st.tabs([
        "ML Models",
        "Data",
        "MFT Blender",
        "Phi-Bot",
        "Plutus Bot",
        "Backtests",
        "System",
    ])

    with tabs[0]: render_ml_status()
    with tabs[1]: render_fetch_data()
    with tabs[2]: render_mft_blender(config)
    with tabs[3]: render_phi_bot(config)
    with tabs[4]: render_plutus_bot(config)
    with tabs[5]: render_backtests(config)
    with tabs[6]: render_system_status()

    _render_footer()

    # Mobile bottom nav (phones/tablets)
    _render_mobile_nav()


if __name__ == "__main__":
    main()

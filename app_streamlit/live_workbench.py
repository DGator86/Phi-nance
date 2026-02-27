#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-nance Live Backtest Workbench
=================================

Premium quant SaaS-style workbench:
- Fetch & cache historical data
- Select & tune indicators
- Blend multiple indicators
- PhiAI auto-tuning
- Equities + Options backtests
- Live progress + reproducible runs
- Dark purple/orange theme

Run:
    python -m streamlit run app_streamlit/live_workbench.py
"""

import os
import sys
import time
import threading
import importlib
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # type: ignore

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("IS_BACKTESTING", "True")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Inject dark theme CSS
_CSS_PATH = _ROOT / ".streamlit" / "styles.css"
if _CSS_PATH.exists():
    with open(_CSS_PATH, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Indicator catalog (maps to strategies)
# ─────────────────────────────────────────────────────────────────────────────
INDICATOR_CATALOG = {
    "RSI": {
        "description": (
            "Relative Strength Index — UP < oversold; DOWN > overbought."
        ),
        "params": {
            "rsi_period": (2, 50, 14),
            "oversold": (10, 50, 30),
            "overbought": (50, 95, 70)
        },
        "strategy": "strategies.rsi.RSIStrategy",
    },
    "MACD": {
        "description": (
            "Moving Average Convergence Divergence — "
            "bullish/bearish crossover."
        ),
        "params": {
            "fast_period": (2, 50, 12),
            "slow_period": (10, 100, 26),
            "signal_period": (2, 30, 9)
        },
        "strategy": "strategies.macd.MACDStrategy",
    },
    "Bollinger": {
        "description": (
            "Bollinger Bands — UP below lower band; DOWN above upper."
        ),
        "params": {"bb_period": (5, 100, 20), "num_std": (1, 4, 2)},
        "strategy": "strategies.bollinger.BollingerBands",
    },
    "Dual SMA": {
        "description": "Golden cross / death cross.",
        "params": {"fast_period": (2, 100, 10), "slow_period": (10, 300, 50)},
        "strategy": "strategies.dual_sma.DualSMACrossover",
    },
    "Mean Reversion": {
        "description": "UP < SMA; DOWN > SMA.",
        "params": {"sma_period": (5, 200, 20)},
        "strategy": "strategies.mean_reversion.MeanReversion",
    },
    "Breakout": {
        "description": "Donchian channel breakout/breakdown.",
        "params": {"channel_period": (5, 100, 20)},
        "strategy": "strategies.breakout.ChannelBreakout",
    },
    "Buy & Hold": {
        "description": "Naive long-only baseline.",
        "params": {},
        "strategy": "strategies.buy_and_hold.BuyAndHold",
    },
}

BLEND_METHODS = [
    "Weighted Sum", "Regime-Weighted", "Voting", "PhiAI Chooses"
]
METRICS = [
    "ROI", "CAGR", "Sharpe", "Max Drawdown", "Direction Accuracy",
    "Profit Factor"
]
EXIT_STRATEGIES = ["Signal exit", "SL/TP", "Trailing stop", "Time exit"]
POSITION_SIZING = ["Fixed %", "Fixed shares"]

# Visual spec — chart colors (purple/orange theme)
CHART_COLORS = ["#a855f7", "#f97316", "#22c55e", "#06b6d4", "#eab308"]


def _load_strategy(module_cls: str):
    """
    Dynamically load a strategy class from a string.
    """
    module_path, cls_name = module_cls.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _av_datasource():
    """Returns the AlphaVantageFixedDataSource class (lazy import)."""
    # pylint: disable=import-outside-toplevel
    from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
    return AlphaVantageFixedDataSource


def _run_backtest(strategy_class, params: dict, config: dict):
    """Execution wrapper for Lumibot backtests."""
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")
    tf = config.get("timeframe", "1D")
    timestep = "day" if tf == "1D" else "minute"
    results, strat = strategy_class.run_backtest(
        datasource_class=_av_datasource(),
        backtesting_start=config["start"],
        backtesting_end=config["end"],
        budget=config["initial_capital"],
        benchmark_asset=config.get("benchmark", "SPY"),
        parameters=params,
        api_key=av_api_key,
        timestep=timestep,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_indicators=False,
        show_progress_bar=False,
        quiet_logs=True,
    )
    return results, strat


def _compute_accuracy(strat):
    """Compute direction accuracy for a strategy (lazy import)."""
    # pylint: disable=import-outside-toplevel
    from strategies.prediction_tracker import compute_prediction_accuracy
    return compute_prediction_accuracy(strat)


def _run_fully_automated(
    symbol: str,
    start_date: str,
    end_date: str,
    capital: float,
    use_ollama: bool,
    ollama_host: str,
):
    """One-shot: pipeline + backtest. Fully automated."""
    progress = st.progress(0, text="Fully automated: starting... 0%")
    result_holder = [None]
    exc_holder = [None]

    def run():
        try:
            # pylint: disable=import-outside-toplevel
            from phi.phiai.auto_pipeline import run_fully_automated as run_pipeline

            # NOTE: Do not call progress.progress() here — Streamlit widgets
            # require the main thread; this runs in a worker thread.
            cfg, indicators, blend_method, explanation, ohlcv = run_pipeline(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
                ollama_host=ollama_host,
                use_ollama=use_ollama,
            )

            # Direct backtest on pipeline OHLCV — bypasses Lumibot/datasource
            blend_weights = {k: 1.0 / len(indicators) for k in indicators}
            from phi.backtest import run_direct_backtest

            results, strat = run_direct_backtest(
                ohlcv=ohlcv,
                symbol=symbol,
                indicators=indicators,
                blend_weights=blend_weights,
                blend_method=blend_method,
                signal_threshold=0.15,
                initial_capital=capital,
            )
            sc = _compute_accuracy(strat) if hasattr(strat, "prediction_log") else {}
            result_holder[0] = (cfg, results, strat, indicators, blend_method, sc, explanation)
        except Exception as e:
            exc_holder[0] = e

    th = threading.Thread(target=run)
    th.start()
    pct = 10
    start_t = time.time()
    while th.is_alive():
        time.sleep(0.4)
        elapsed = time.time() - start_t
        pct = min(95, 10 + int(elapsed * 1.2))
        progress.progress(pct / 100, text=f"Fully automated... {pct}%")

    if exc_holder[0]:
        progress.empty()
        st.error(str(exc_holder[0]))
        st.exception(exc_holder[0])
        return

    progress.progress(1.0, text="Complete — 100%")
    time.sleep(0.5)
    progress.empty()

    cfg, results, strat, indicators, blend_method, sc, explanation = result_holder[0]
    blend_weights = {k: 1.0 / len(indicators) for k in indicators}

    with st.expander("AI decisions", expanded=True):
        st.text(explanation)

    _display_results(cfg, results, strat, indicators, blend_method, blend_weights, sc)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────
def render_dataset_builder():
    """Render Step 1: Data fetching and caching UI."""
    st.markdown("### Step 1 — Dataset Builder")

    col_mode, col_sym, col_range = st.columns([1, 2, 2])
    with col_mode:
        trading_mode = st.selectbox(
            "Trading Mode", ["Equities", "Options"], key="ds_mode"
        )
    with col_sym:
        symbols_raw = st.text_input("Symbol(s)", value="SPY", key="ds_symbols",
                                    help="Comma-separated: SPY, QQQ, AAPL")
    with col_range:
        start_d = st.date_input(
            "Start", value=date(2020, 1, 1), key="ds_start"
        )
        end_d = st.date_input(
            "End", value=date(2024, 12, 31), key="ds_end"
        )

    col_tf, col_vendor, col_cap = st.columns(3)
    with col_tf:
        timeframe = st.selectbox(
            "Timeframe", ["1D", "4H", "1H", "15m", "5m", "1m"], key="ds_tf"
        )
    with col_vendor:
        vendor = st.selectbox(
            "Data Vendor",
            ["Alpha Vantage", "yfinance", "Binance Public"],
            key="ds_vendor"
        )
    with col_cap:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=100_000,
            min_value=1_000,
            step=10_000,
            key="ds_cap",
            help="Starting capital for backtest",
        )

    if initial_capital <= 0:
        st.error("Initial capital must be > 0")
        return None

    col_fetch, col_use = st.columns(2)
    with col_fetch:
        fetch_clicked = st.button(
            "Fetch & Cache Data", type="primary", key="ds_fetch"
        )
    with col_use:
        use_cached = st.button("Use Cached Data", key="ds_use")

    vendor_map = {
        "Alpha Vantage": "alphavantage",
        "yfinance": "yfinance",
        "Binance Public": "binance_public"
    }
    vendor_key = vendor_map.get(vendor, "alphavantage")
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        st.warning("Enter at least one symbol.")
        return None

    dfs = {}

    if fetch_clicked or use_cached:
        # pylint: disable=import-outside-toplevel
        from phi.data import fetch_and_cache, get_cached_dataset

        with st.status("Loading data...", expanded=True) as s:
            for sym in symbols:
                try:
                    if fetch_clicked:
                        df = fetch_and_cache(
                            vendor_key, sym, timeframe,
                            str(start_d), str(end_d),
                        )
                    else:
                        df = get_cached_dataset(
                            vendor_key, sym, timeframe,
                            str(start_d), str(end_d),
                        )
                    if df is not None and not df.empty:
                        dfs[sym] = df
                except Exception as e:  # pylint: disable=broad-except
                    st.error(f"{sym}: {e}")
            if dfs:
                st.session_state["workbench_dataset"] = dfs
                st.session_state["workbench_config"] = {
                    "trading_mode": trading_mode.lower(),
                    "symbols": symbols,
                    "start": datetime.combine(start_d, datetime.min.time()),
                    "end": datetime.combine(end_d, datetime.min.time()),
                    "timeframe": timeframe,
                    "vendor": vendor_key,
                    "initial_capital": float(initial_capital),
                    "benchmark": symbols[0],
                }
                bars_count = sum(len(d) for d in dfs.values())
                s.update(
                    label=f"Cached {bars_count:,} bars",
                    state="complete"
                )
                s.update(
                    label=f"Cached {bars_count:,} bars",
                    state="complete"
                )
            else:
                s.update(label="No data", state="error")

    if st.session_state.get("workbench_dataset"):
        dfs = st.session_state["workbench_dataset"]
        cfg = st.session_state.get("workbench_config", {})
        st.success(
            f"**Dataset ready:** {', '.join(dfs.keys())} · "
            f"{sum(len(d) for d in dfs.values()):,} bars · "
            f"${cfg.get('initial_capital', 0):,.0f} initial capital"
        )
        for sym, df in list(dfs.items())[:3]:
            with st.expander(f"{sym} — {len(df):,} rows"):
                st.dataframe(df.tail(20), width="stretch")
        return cfg
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Indicator Selection
# ─────────────────────────────────────────────────────────────────────────────
def render_indicator_selection():
    """Render Step 2: Strategy indicator selection and manual tuning."""
    st.markdown("### Step 2 — Indicator Selection")

    selected = st.session_state.get("workbench_indicators", {})

    left, _ = st.columns([1, 1])
    with left:
        search = st.text_input(
            "Search indicators", key="ind_search", placeholder="RSI, MACD..."
        )
        available = [
            k for k in INDICATOR_CATALOG
            if not search or search.lower() in k.lower()
        ]
        for name in available:
            info = INDICATOR_CATALOG[name]
            enabled = st.checkbox(
                f"**{name}** — {info['description'][:50]}...",
                value=name in selected,
                key=f"ind_{name}"
            )
            if enabled:
                if name not in selected:
                    selected[name] = {
                        "enabled": True, "auto_tune": False, "params": {}
                    }
                selected[name]["enabled"] = True
                selected[name]["auto_tune"] = st.toggle(
                    "PhiAI Auto-tune",
                    value=selected[name].get("auto_tune", False),
                    key=f"at_{name}"
                )
                with st.expander("Manual tuning", expanded=False):
                    for pname, (lo, hi, default) in info["params"].items():
                        selected[name]["params"][pname] = st.slider(
                            pname, lo, hi, default, key=f"param_{name}_{pname}"
                        )
            else:
                if name in selected:
                    del selected[name]

    st.session_state["workbench_indicators"] = selected
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Blending Panel
# ─────────────────────────────────────────────────────────────────────────────
def render_blending(indicators: dict):
    """
    Render the blending panel for multiple indicators.
    """
    if len(indicators) < 2:
        st.caption("Select 2+ indicators to enable blending.")
        return "weighted_sum", {}

    st.markdown("### Step 3 — Blending Panel")
    method = st.selectbox("Blend Mode", BLEND_METHODS, key="blend_method")
    method_map = {
        "Weighted Sum": "weighted_sum",
        "Regime-Weighted": "regime_weighted",
        "Voting": "voting",
        "PhiAI Chooses": "phiai_chooses"
    }
    method_key = method_map.get(method, "weighted_sum")

    weights = {}
    for name in indicators:
        weights[name] = st.slider(
            f"Weight: {name}", 0.0, 1.0, 1.0 / len(indicators), 0.05,
            key=f"wt_{name}"
        )
    return method_key, weights


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — PhiAI Panel
# ─────────────────────────────────────────────────────────────────────────────
def render_phiai():
    """
    Render the PhiAI panel for automated optimization.
    """
    st.markdown("### Step 4 — PhiAI Panel")
    phiai_full = st.toggle(
        "PhiAI Full Auto", value=False, key="phiai_full",
        help="Auto-enable/disable indicators, tune params, select blend"
    )
    if phiai_full:
        st.info(
            "PhiAI will optimize indicators, parameters, and blend. "
            "Regime-aware adjustments applied."
        )
        st.number_input("Max indicators", 1, 10, 5, key="phiai_max")
        st.checkbox("No shorting", value=True, key="phiai_noshort")
    return phiai_full


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Backtest Controls
# ─────────────────────────────────────────────────────────────────────────────
def render_backtest_controls(config: dict):
    """
    Render the backtest control panel based on trading mode.
    """
    if not config:
        return {}

    st.markdown("### Step 5 — Backtest Controls")
    mode = config.get("trading_mode", "equities")

    if mode == "equities":
        allow_short = st.checkbox(
            "Allow shorting", value=False, key="bt_short"
        )
        pos_sizing = st.selectbox(
            "Position sizing", POSITION_SIZING, key="bt_pos"
        )
        exit_strat = st.selectbox(
            "Exit strategy", EXIT_STRATEGIES, key="bt_exit"
        )
        return {
            "allow_short": allow_short,
            "position_sizing": pos_sizing,
            "exit_strategy": exit_strat
        }
    else:
        st.caption("Options mode: Long Call/Put with delta-based simulation.")
        strat_type = st.selectbox(
            "Strategy", ["long_call", "long_put"], key="opt_strat"
        )
        exit_profit = st.slider(
            "Exit profit %", 0.2, 1.0, 0.5, 0.1, key="opt_exit_profit"
        )
        exit_stop = st.slider(
            "Exit stop %", -0.5, -0.1, -0.3, 0.05, key="opt_exit_stop"
        )
        opts = {
            "strategy_type": strat_type,
            "exit_profit_pct": exit_profit,
            "exit_stop_pct": exit_stop
        }
        st.session_state["bt_options_controls"] = opts
        return opts


# ─────────────────────────────────────────────────────────────────────────────
# Run & Results
# ─────────────────────────────────────────────────────────────────────────────
def render_run_and_results(
    config: dict, indicators: dict, blend_method: str, blend_weights: dict
):
    """
    Orchestrate the backtest execution and result display.
    """
    if not config or not indicators:
        st.info("Complete Steps 1–2 to run a backtest.")
        return

    st.markdown("---")
    st.markdown("## Run Backtest")

    st.selectbox("Primary metric", METRICS, key="primary_metric")
    col_run, _ = st.columns(2)
    with col_run:
        run_clicked = st.button("Run Backtest", type="primary", key="run_bt")

    phiai_full = st.session_state.get("phiai_full", False)
    trading_mode = config.get("trading_mode", "equities")

    if run_clicked:
        indicators_to_use = dict(indicators)
        phiai_explanation = ""

        # PhiAI optimization when enabled
        workbench_data = st.session_state.get("workbench_dataset") or {}
        if phiai_full and workbench_data:
            phiai_progress = st.progress(0, text="PhiAI optimizing... 0%")
            phiai_result = [None]
            phiai_exc = [None]
            _sym = config["symbols"][0]
            _ohlcv = workbench_data.get(_sym)

            def run_phiai():
                try:
                    if _ohlcv is not None and len(_ohlcv) > 100:
                        # pylint: disable=import-outside-toplevel
                        from phi.phiai import run_phiai_optimization
                        phiai_result[0] = run_phiai_optimization(
                            _ohlcv, indicators_to_use, max_iter_per_indicator=15
                        )
                except Exception as ex:  # pylint: disable=broad-except
                    phiai_exc[0] = ex

            th_phiai = threading.Thread(target=run_phiai)
            th_phiai.start()
            pct = 0
            start_t = time.time()
            while th_phiai.is_alive():
                time.sleep(0.3)
                elapsed = time.time() - start_t
                pct = min(95, int(elapsed * 12))  # ~8s to 95%
                phiai_progress.progress(
                    pct / 100,
                    text=f"PhiAI optimizing... {pct}%"
                )
            if phiai_exc[0] is not None:
                st.warning(f"PhiAI optimization skipped: {phiai_exc[0]}")
            elif (phiai_res := phiai_result[0]) is not None:
                assert isinstance(phiai_res, (list, tuple))
                if len(phiai_res) == 2:
                    # pylint: disable=unpacking-non-sequence
                    indicators_to_use, phiai_explanation = phiai_res
                    st.session_state["phiai_explanation"] = phiai_explanation
            phiai_progress.progress(1.0, text="PhiAI complete — 100%")
            time.sleep(0.3)
            phiai_progress.empty()

        # Options mode: use phi.options.backtest
        if trading_mode == "options":
            try:
                opt_progress = st.progress(
                    0, text="Running options backtest... 0%"
                )
                opt_result = [None]
                opt_exc = [None]

                def run_opt():
                    try:
                        dfs = st.session_state.get("workbench_dataset", {})
                        sym = config["symbols"][0]
                        ohlcv = dfs.get(sym)
                        if ohlcv is None or ohlcv.empty:
                            msg = "No data for options backtest. Fetch first."
                            raise ValueError(msg)

                        bt_opts = st.session_state.get(
                            "bt_options_controls", {}
                        )
                        # pylint: disable=import-outside-toplevel
                        from phi.options import run_options_backtest
                        opt_result[0] = run_options_backtest(
                            ohlcv,
                            symbol=sym,
                            strategy_type=bt_opts.get(
                                "strategy_type", "long_call"
                            ),
                            initial_capital=config.get(
                                "initial_capital", 100_000
                            ),
                            position_pct=0.1,
                            exit_profit_pct=bt_opts.get(
                                "exit_profit_pct", 0.5
                            ),
                            exit_stop_pct=bt_opts.get(
                                "exit_stop_pct", -0.3
                            ),
                        )
                    except Exception as e:  # pylint: disable=broad-except
                        opt_exc[0] = e

                th_opt = threading.Thread(target=run_opt)
                th_opt.start()
                pct = 0
                start_t = time.time()
                while th_opt.is_alive():
                    time.sleep(0.2)
                    elapsed = time.time() - start_t
                    pct = min(95, int(elapsed * 25))  # ramps quickly
                    opt_progress.progress(
                        pct / 100,
                        text=f"Running options backtest... {pct}%"
                    )
                if (oe := opt_exc[0]) is not None:
                    opt_progress.empty()
                    # pylint: disable=raising-bad-type
                    raise oe
                results = opt_result[0]
                opt_progress.progress(1.0, text="Complete — 100%")
                time.sleep(0.3)
                opt_progress.empty()
                if results:
                    _display_results(
                        config, results, None, indicators_to_use,
                        blend_method, blend_weights
                    )
                else:
                    st.error("Options backtest returned no results.")
            except Exception as e:  # pylint: disable=broad-exception-caught
                st.error(str(e))
                st.exception(e)
            return

        # Equities: direct vectorized backtest (no Lumibot dependency)
        progress = st.progress(0, text="Preparing backtest... 0%")

        try:
            sym = config["symbols"][0]
            dfs = st.session_state.get("workbench_dataset", {})
            ohlcv = dfs.get(sym)
            if ohlcv is None or (hasattr(ohlcv, "empty") and ohlcv.empty):
                progress.empty()
                st.error(
                    "No dataset loaded. Complete Step 1 (Fetch & Cache Data) first."
                )
                return

            # Run direct backtest in thread with live progress %
            result_holder = [None]
            exc_holder = [None]

            def run_bt():
                try:
                    # pylint: disable=import-outside-toplevel
                    from phi.backtest import run_direct_backtest
                    result_holder[0] = run_direct_backtest(
                        ohlcv=ohlcv,
                        symbol=sym,
                        indicators=indicators_to_use,
                        blend_weights=blend_weights,
                        blend_method=blend_method,
                        signal_threshold=0.15,
                        initial_capital=config["initial_capital"],
                    )
                except Exception as e:  # pylint: disable=broad-except
                    exc_holder[0] = e

            th = threading.Thread(target=run_bt)
            th.start()

            pct = 5
            start_t = time.time()
            while th.is_alive():
                time.sleep(0.3)
                elapsed = time.time() - start_t
                pct = min(95, 5 + int(elapsed * 8))
                progress.progress(
                    pct / 100, text=f"Running backtest... {pct}%"
                )

            if (eh := exc_holder[0]) is not None:
                # pylint: disable=raising-bad-type
                raise eh

            if (res_h := result_holder[0]) is not None:
                results, strat = res_h
                progress.progress(1.0, text="Complete — 100%")
                time.sleep(0.3)
                progress.empty()

                has_log = hasattr(strat, "prediction_log")
                sc = _compute_accuracy(strat) if has_log else {}
                _display_results(
                    config, results, strat, indicators_to_use,
                    blend_method, blend_weights, sc
                )
            else:
                progress.empty()
                st.error("Backtest returned no results or failed silently.")

            if phiai_full and st.session_state.get("phiai_explanation"):
                with st.expander("PhiAI changes"):
                    st.text(st.session_state["phiai_explanation"])

        except Exception as e:  # pylint: disable=broad-except
            progress.empty()
            st.error(str(e))
            st.exception(e)


def _extract_scalar(val):
    """
    Extract a scalar float from a value that may be a dict.
    (lumibot wraps some metrics in dictionaries).
    """
    if isinstance(val, dict):
        # lumibot may return {"drawdown": float, ...} or {"value": float, ...}
        for key in ("drawdown", "value", "max_drawdown", "return"):
            if key in val:
                return val[key]
        # fallback: first numeric value found
        for v in val.values():
            if isinstance(v, (int, float)):
                return v
        return None
    return val


def _display_results(
    config, results, strat, indicators, blend_method, blend_weights, sc=None
):
    """
    Render backtest metrics, charts, and trade logs to the Streamlit UI.
    """
    sc = sc or {}
    tr = _extract_scalar(
        getattr(results, "total_return", None) or results.get("total_return")
    )
    cagr = _extract_scalar(
        getattr(results, "cagr", None) or results.get("cagr")
    )
    dd = _extract_scalar(
        getattr(results, "max_drawdown", None) or results.get("max_drawdown")
    )
    sharpe = _extract_scalar(
        getattr(results, "sharpe", None) or results.get("sharpe")
    )
    cap = config.get("initial_capital", 100_000)
    pv = getattr(results, "portfolio_value", None)
    if pv is None:
        pv = results.get("portfolio_value", [])
    end_cap = pv[-1] if pv and len(pv) else cap
    net_pl = end_cap - cap
    net_pct = (net_pl / cap) * 100 if cap else 0

    st.markdown("### Results")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Start Capital", f"${cap:,.0f}")
    r2.metric("End Capital", f"${end_cap:,.0f}")
    r3.metric("Net P/L", f"${net_pl:+,.0f}", f"{net_pct:+.1f}%")
    r4.metric(
        "CAGR", f"{cagr:+.1%}" if isinstance(cagr, (int, float)) else "—"
    )
    r5.metric(
        "Sharpe", f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "—"
    )

    tab_sum, tab_curve, tab_trades, tab_metrics = st.tabs(
        ["Summary", "Equity Curve", "Trades", "Metrics"]
    )
    with tab_sum:
        st.metric(
            "Max Drawdown",
            f"{dd:.1%}" if isinstance(dd, (int, float)) else "—"
        )
        acc = sc.get('accuracy', 0)
        acc_text = (
            f"{acc:.1%}" if sc and isinstance(acc, (int, float)) else "—"
        )
        st.metric("Direction Accuracy", acc_text)
    with tab_curve:
        if pv and len(pv) > 1:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    y=pv, mode="lines",
                    line=dict(color=CHART_COLORS[0], width=2)
                )
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f0f12",
                plot_bgcolor="#1a1a1f",
                font_color="#e4e4e7",
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
    with tab_trades:
        if strat and hasattr(strat, "prediction_log") and strat.prediction_log:
            st.dataframe(pd.DataFrame(strat.prediction_log), width="stretch")
        else:
            st.caption("No trade log.")
    with tab_metrics:
        payload = {
            "total_return": tr, "cagr": cagr, "max_drawdown": dd,
            "sharpe": sharpe, "accuracy": sc.get("accuracy")
        }
        if isinstance(results, dict) and results.get("options_snapshot"):
            payload["options_snapshot"] = results.get("options_snapshot")
        st.json(payload)

    from phi.run_config import RunConfig, RunHistory
    run_cfg = RunConfig(
        symbols=config["symbols"],
        start_date=str(config["start"].date()),
        end_date=str(config["end"].date()),
        timeframe=config.get("timeframe", "1D"),
        initial_capital=cap,
        indicators=indicators,
        blend_method=blend_method,
        blend_weights=blend_weights,
    )
    hist = RunHistory()
    run_id = hist.create_run(run_cfg)
    hist.save_results(run_id, {
        "total_return": tr, "cagr": cagr, "max_drawdown": dd,
        "sharpe": sharpe, "accuracy": sc.get("accuracy"), "net_pl": net_pl
    })
    st.caption(f"Run saved: {run_id}")


# ─────────────────────────────────────────────────────────────────────────────
# AI Agents (Ollama)
# ─────────────────────────────────────────────────────────────────────────────
def render_ai_agents():
    """Ollama-powered free AI agents for regime/strategy Q&A."""
    st.markdown("### AI Agents (Ollama)")
    st.caption(
        "Free local models via [Ollama](https://ollama.com). "
        "Install: [ollama.com/download](https://ollama.com/download) · "
        "Then: `ollama pull llama3.2` or `ollama pull 0xroyce/plutus`"
    )

    host = st.text_input(
        "Ollama Host",
        value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        key="ollama_host",
        help="Default: http://localhost:11434",
    )

    col_check, col_list = st.columns(2)
    with col_check:
        if st.button("Check connection"):
            # pylint: disable=import-outside-toplevel
            from phi.agents import check_ollama_ready
            ok = check_ollama_ready(host)
            if ok:
                st.success("Ollama is running")
            else:
                st.error(
                    "Ollama not reachable. Install from ollama.com and run it."
                )
    with col_list:
        if st.button("List models"):
            # pylint: disable=import-outside-toplevel
            from phi.agents import list_ollama_models
            models = list_ollama_models(host)
            if models:
                names = [
                    m.get("name", "").split(":")[0]
                    for m in models if m.get("name")
                ]
                st.session_state["ollama_models"] = list(dict.fromkeys(names))
                st.success(
                    f"Found {len(st.session_state['ollama_models'])} model(s)"
                )
            else:
                msg = (
                    "No models or Ollama not running. "
                    "Run: ollama pull llama3.2"
                )
                st.warning(msg)

    m_def = ["llama3.2", "0xroyce/plutus"]
    model_list = st.session_state.get("ollama_models", m_def)
    model = st.selectbox(
        "Model",
        options=model_list,
        index=0 if model_list else 0,
        key="ollama_model",
        help="Pull with: ollama pull <model>",
    )

    prompt = st.text_area(
        "Ask AI (regime, strategy, indicators, etc.)",
        placeholder=(
            "e.g. What does RSI oversold mean? "
            "When to use regime-weighted blending?"
        ),
        key="ollama_prompt",
        height=80,
    )
    if st.button("Send", key="ollama_send") and prompt:
        with st.spinner("Thinking..."):
            try:
                from phi.agents import OllamaAgent
                agent = OllamaAgent(model=model, host=host)
                reply = agent.chat(
                    prompt,
                    system=(
                        "You are a quantitative trading assistant. "
                        "Be concise."
                    ),
                )
                st.markdown("**Reply:**")
                st.markdown(reply)
            except Exception as e:
                st.error(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Options Workbench
# ─────────────────────────────────────────────────────────────────────────────

# Strategy display names (imported from engine at render time to avoid top-level import)
_OW_STRATEGY_LABELS = [
    "Long Call",
    "Long Put",
    "Covered Call",
    "Cash-Secured Put",
    "Bull Call Spread",
    "Bear Put Spread",
    "Straddle",
    "Strangle",
    "Iron Condor",
]
_OW_STRATEGY_KEYS = [
    "long_call",
    "long_put",
    "covered_call",
    "cash_secured_put",
    "bull_call_spread",
    "bear_put_spread",
    "straddle",
    "strangle",
    "iron_condor",
]
_OW_STRATEGY_MAP = dict(zip(_OW_STRATEGY_LABELS, _OW_STRATEGY_KEYS))

_OW_STRATEGY_TOOLTIPS = {
    "Long Call":        "Buy an ATM call. Profits from upward moves. Loss capped at premium.",
    "Long Put":         "Buy an ATM put. Profits from downward moves. Loss capped at premium.",
    "Covered Call":     "Hold 100 shares + sell OTM call. Generates income; caps upside.",
    "Cash-Secured Put": "Sell OTM put backed by full cash reserve. Collects premium; capped downside.",
    "Bull Call Spread": "Buy ATM call, sell OTM call. Lower cost; capped profit and loss.",
    "Bear Put Spread":  "Buy ATM put, sell OTM put. Lower cost bearish play; capped P&L.",
    "Straddle":         "Buy ATM call + put. Profits from large moves in either direction.",
    "Strangle":         "Buy OTM call + OTM put. Cheaper than straddle; needs bigger move.",
    "Iron Condor":      "Sell OTM strangle, buy wider strangle. Profits in low-volatility range.",
}


def _ow_compute_signal(
    ohlcv: pd.DataFrame,
    indicator_names: list,
) -> pd.Series:
    """Compute blended entry signal from selected indicators on the OHLCV."""
    # pylint: disable=import-outside-toplevel
    from phi.indicators.simple import compute_indicator
    from phi.blending import blend_signals

    signals: dict = {}
    for name in indicator_names:
        info = INDICATOR_CATALOG.get(name, {})
        defaults = {k: default for k, (_, _, default) in info.get("params", {}).items()}
        sig = compute_indicator(name, ohlcv, defaults)
        if sig is not None and not sig.empty:
            signals[name] = sig

    if not signals:
        return pd.Series(0.0, index=ohlcv.index)

    sig_df = pd.DataFrame(signals).reindex(ohlcv.index).ffill().bfill()
    return blend_signals(sig_df, method="weighted_sum")


def _ow_display_results(results: dict, initial_capital: float) -> None:
    """Render options workbench results: metrics, charts, trade log."""
    if not results or not results.get("portfolio_value"):
        st.error("No results to display.")
        return

    pv = results["portfolio_value"]
    end_cap = pv[-1] if pv else initial_capital
    net_pl = end_cap - initial_capital
    net_pct = (net_pl / initial_capital * 100) if initial_capital else 0

    # ── Top metric strip ─────────────────────────────────────────────────────
    st.markdown("### Options Backtest Results")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Start Capital",  f"${initial_capital:,.0f}")
    m2.metric("End Capital",    f"${end_cap:,.0f}")
    m3.metric("Net P/L",        f"${net_pl:+,.0f}", f"{net_pct:+.1f}%")
    cagr = results.get("cagr", 0)
    m4.metric("CAGR",   f"{cagr:+.1%}" if isinstance(cagr, float) else "—")
    sharpe = results.get("sharpe", 0)
    m5.metric("Sharpe", f"{sharpe:.2f}" if isinstance(sharpe, float) else "—")
    dd = results.get("max_drawdown", 0)
    m6.metric("Max DD", f"{dd:.1%}" if isinstance(dd, float) else "—")

    # ── Trade stats strip ────────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    n_trades = results.get("total_trades", 0)
    s1.metric("Total Trades",  str(n_trades))
    wr = results.get("win_rate", 0)
    s2.metric("Win Rate",      f"{wr:.1%}" if n_trades else "—")
    pf = results.get("profit_factor", 0)
    s3.metric("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")
    aw = results.get("avg_win", 0)
    s4.metric("Avg Win $",     f"${aw:+,.0f}" if n_trades else "—")
    al = results.get("avg_loss", 0)
    s5.metric("Avg Loss $",    f"${al:+,.0f}" if n_trades else "—")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_curve, tab_trades, tab_pnl_dist, tab_iv, tab_metrics = st.tabs([
        "Equity Curve", "Trade Log", "P&L Distribution", "IV History", "Full Metrics"
    ])

    with tab_curve:
        if len(pv) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=pv, mode="lines",
                line=dict(color=CHART_COLORS[0], width=2),
                name="Portfolio Value",
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ))
            # Shade drawdown
            pv_arr = pd.Series(pv)
            rolling_max = pv_arr.cummax()
            dd_arr = (pv_arr - rolling_max) / rolling_max.clip(lower=1e-8)
            fig.add_trace(go.Scatter(
                y=rolling_max, mode="lines",
                line=dict(color=CHART_COLORS[2], width=1, dash="dot"),
                name="Rolling High",
                opacity=0.5,
                hovertemplate="$%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f0f12", plot_bgcolor="#1a1a1f",
                font_color="#e4e4e7",
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Not enough data for equity curve.")

    with tab_trades:
        trade_log = results.get("trade_log")
        if trade_log is not None and not trade_log.empty:
            # Color-code P&L column
            st.dataframe(trade_log, use_container_width=True)
            csv = trade_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trade Log (CSV)", csv, "options_trades.csv", "text/csv"
            )
        else:
            st.caption("No trades recorded. Check signal threshold, DTE, and position sizing.")

    with tab_pnl_dist:
        pnls = results.get("pnls", [])
        if pnls:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=pnls,
                nbinsx=min(30, max(10, len(pnls) // 3)),
                marker_color=CHART_COLORS[1],
                name="P&L per Trade ($)",
            ))
            fig2.add_vline(x=0, line_color="#e4e4e7", line_dash="dash")
            fig2.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f0f12", plot_bgcolor="#1a1a1f",
                font_color="#e4e4e7",
                xaxis_title="Trade P&L ($)",
                yaxis_title="Count",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Waterfall for cumulative P&L
            cumulative = pd.Series(pnls).cumsum().tolist()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                y=cumulative, mode="lines+markers",
                line=dict(color=CHART_COLORS[3], width=2),
                marker=dict(
                    color=[CHART_COLORS[2] if p >= 0 else CHART_COLORS[1] for p in pnls],
                    size=5,
                ),
                name="Cumulative P&L",
            ))
            fig3.add_hline(y=0, line_color="#e4e4e7", line_dash="dot")
            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f0f12", plot_bgcolor="#1a1a1f",
                font_color="#e4e4e7",
                yaxis_title="Cumulative P&L ($)",
                xaxis_title="Trade #",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.caption("No P&L data.")

    with tab_iv:
        iv_series = results.get("iv_series", [])
        if iv_series:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                y=[v * 100 for v in iv_series], mode="lines",
                line=dict(color=CHART_COLORS[4], width=1),
                name="Realized IV × Factor (%)",
            ))
            fig4.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f0f12", plot_bgcolor="#1a1a1f",
                font_color="#e4e4e7",
                yaxis_title="IV (%)",
                xaxis_title="Bar",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.caption("No IV data.")

    with tab_metrics:
        payload = {
            "total_return":    results.get("total_return"),
            "cagr":            results.get("cagr"),
            "max_drawdown":    results.get("max_drawdown"),
            "sharpe":          results.get("sharpe"),
            "total_trades":    results.get("total_trades"),
            "win_rate":        results.get("win_rate"),
            "profit_factor":   results.get("profit_factor"),
            "avg_win":         results.get("avg_win"),
            "avg_loss":        results.get("avg_loss"),
            "gross_win":       results.get("gross_win"),
            "gross_loss":      results.get("gross_loss"),
            "max_consec_loss": results.get("max_consec_loss"),
        }
        st.json({k: (round(v, 6) if isinstance(v, float) else v) for k, v in payload.items()})


def render_options_workbench() -> None:
    """
    Full options strategy testing bench.

    Uses the dataset already loaded in Step 1 (workbench_dataset) when
    available. Falls back to an inline symbol/date fetch otherwise.
    Supports 9 strategy types with Black-Scholes pricing, 4 exit rules,
    signal-driven or periodic entry, and full result visualizations.
    """
    st.markdown("### Options Workbench")

    # ── Dataset source ────────────────────────────────────────────────────────
    dfs = st.session_state.get("workbench_dataset", {})
    cfg = st.session_state.get("workbench_config", {})

    if dfs:
        sym_choices = list(dfs.keys())
        st.success(
            f"Using dataset from Step 1: {', '.join(sym_choices)}"
        )
        ow_sym = st.selectbox("Symbol", sym_choices, key="ow_sym_picker")
        ohlcv_source = dfs.get(ow_sym)
        initial_capital = float(cfg.get("initial_capital", 100_000))
    else:
        st.info(
            "No dataset loaded yet. "
            "Complete Step 1 above, or fetch inline below."
        )
        col_s, col_d1, col_d2, col_cap = st.columns([1, 1, 1, 1])
        with col_s:
            ow_sym = st.text_input("Symbol", value="SPY", key="ow_sym_inline")
        with col_d1:
            ow_start = st.date_input("Start", value=date(2020, 1, 1), key="ow_start_inline")
        with col_d2:
            ow_end = st.date_input("End", value=date(2024, 12, 31), key="ow_end_inline")
        with col_cap:
            initial_capital = float(st.number_input(
                "Capital ($)", value=100_000, min_value=1_000, step=10_000, key="ow_cap_inline"
            ))

        if st.button("Fetch Data for Options", key="ow_fetch_inline"):
            with st.spinner("Fetching..."):
                try:
                    # pylint: disable=import-outside-toplevel
                    from phi.data import fetch_and_cache
                    df_fetched = fetch_and_cache(
                        "alphavantage", ow_sym, "1D",
                        str(ow_start), str(ow_end)
                    )
                    if df_fetched is not None and not df_fetched.empty:
                        st.session_state["ow_inline_dataset"] = {
                            ow_sym: df_fetched
                        }
                        st.success(f"Fetched {len(df_fetched):,} bars for {ow_sym}.")
                    else:
                        st.error("No data returned.")
                except Exception as e:  # pylint: disable=broad-except
                    st.error(str(e))

        inline_dfs = st.session_state.get("ow_inline_dataset", {})
        ohlcv_source = inline_dfs.get(ow_sym)

    if ohlcv_source is None or (hasattr(ohlcv_source, "empty") and ohlcv_source.empty):
        st.caption("Load a dataset above to configure the workbench.")
        return

    st.markdown("---")

    # ── Strategy type ─────────────────────────────────────────────────────────
    strat_col, entry_col = st.columns([2, 2])
    with strat_col:
        strat_label = st.selectbox(
            "Strategy Type", _OW_STRATEGY_LABELS, key="ow_strategy",
            help="Select the options structure to simulate."
        )
        strategy_type = _OW_STRATEGY_MAP[strat_label]
        st.caption(_OW_STRATEGY_TOOLTIPS.get(strat_label, ""))
    with entry_col:
        entry_mode_label = st.selectbox(
            "Entry Mode",
            ["Signal-Driven", "Periodic"],
            key="ow_entry_mode",
            help="Signal-Driven uses indicator signals; Periodic enters every N bars.",
        )
        entry_mode = "signal" if entry_mode_label == "Signal-Driven" else "periodic"

    st.markdown("---")

    # ── Entry parameters ──────────────────────────────────────────────────────
    if entry_mode == "signal":
        st.markdown("**Entry Signal**")
        ind_col, thresh_col = st.columns([3, 1])
        with ind_col:
            ind_names = st.multiselect(
                "Indicators for Entry Signal",
                list(INDICATOR_CATALOG.keys()),
                default=["RSI", "MACD"],
                key="ow_signal_inds",
                help="Composite of selected indicators determines entry timing.",
            )
        with thresh_col:
            sig_threshold = st.slider(
                "Threshold", 0.05, 0.5, 0.15, 0.01, key="ow_sig_thresh",
                help="Minimum signal magnitude to trigger entry.",
            )
        periodic_days = 21
    else:
        p_col, _ = st.columns([1, 3])
        with p_col:
            periodic_days = st.slider(
                "Enter Every N Bars", 5, 90, 21, key="ow_periodic",
                help="Open a new position every N price bars.",
            )
        ind_names = []
        sig_threshold = 0.0

    st.markdown("---")

    # ── Option parameters ─────────────────────────────────────────────────────
    st.markdown("**Option Parameters**")
    op1, op2, op3, op4 = st.columns(4)
    with op1:
        dte = st.slider(
            "DTE at Entry (days)", 7, 120, 30, key="ow_dte",
            help="Days to expiration when the position is opened.",
        )
    with op2:
        iv_factor = st.slider(
            "IV Factor (× realized vol)", 0.5, 3.0, 1.0, 0.05, key="ow_iv_factor",
            help="Multiplier on historical realized vol to estimate implied vol.",
        )
    with op3:
        otm_pct_int = st.slider(
            "OTM % (spreads / condor)", 1, 20, 5, key="ow_otm_pct",
            help="OTM % for short strikes in spread and condor strategies.",
        )
        otm_pct = otm_pct_int / 100.0
    with op4:
        rfr = st.slider(
            "Risk-Free Rate %", 0.0, 10.0, 4.5, 0.1, key="ow_rfr",
            help="Annualized risk-free rate used in Black-Scholes pricing.",
        ) / 100.0

    iv_lb_col, _ = st.columns([1, 3])
    with iv_lb_col:
        iv_lookback = st.slider(
            "IV Lookback (bars)", 5, 63, 21, key="ow_iv_lb",
            help="Number of bars used to compute realized volatility.",
        )

    st.markdown("---")

    # ── Position sizing ───────────────────────────────────────────────────────
    st.markdown("**Position Sizing**")
    ps1, ps2 = st.columns(2)
    with ps1:
        pos_pct = st.slider(
            "Position Size (% of Capital)", 1, 50, 5, key="ow_pos_pct",
            help=(
                "Budget per trade = pos_pct × current capital. "
                "Covered Call / Cash-Secured Put require larger allocations "
                "(10–30%) to fund share/cash reserves."
            ),
        ) / 100.0
    with ps2:
        max_trades = st.slider(
            "Max Concurrent Positions", 1, 10, 3, key="ow_max_trades",
            help="Maximum number of simultaneously open option positions.",
        )

    if strategy_type in ("covered_call", "cash_secured_put") and pos_pct < 0.10:
        st.warning(
            f"**{strat_label}** typically requires 10–30% position sizing "
            "to fund share/cash reserves. Increase 'Position Size' above."
        )

    st.markdown("---")

    # ── Exit rules ────────────────────────────────────────────────────────────
    st.markdown("**Exit Rules**")
    hold_to_expiry = st.toggle(
        "Hold to Expiry (ignore all other exit rules)",
        value=False, key="ow_hold_expiry"
    )

    if not hold_to_expiry:
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            profit_target = st.slider(
                "Profit Target (% of allocated)", 10, 300, 50, key="ow_profit",
                help="Close when unrealized P&L reaches this % of allocated capital.",
            ) / 100.0
        with ex2:
            stop_loss = st.slider(
                "Stop Loss (% of allocated)", 10, 100, 50, key="ow_stop",
                help="Close when loss reaches this % of allocated capital.",
            ) / 100.0
        with ex3:
            dte_exit = st.slider(
                "DTE Exit Threshold (days)", 0, 30, 5, key="ow_dte_exit",
                help="Close when remaining DTE drops to or below this value.",
            )
    else:
        profit_target = 1.0
        stop_loss = 1.0
        dte_exit = 0

    st.markdown("---")

    # ── Run button ────────────────────────────────────────────────────────────
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_clicked = st.button(
            "Run Options Backtest", type="primary", key="ow_run"
        )

    if not run_clicked:
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    progress = st.progress(0, text="Computing options backtest... 0%")
    result_holder: list = [None]
    exc_holder: list = [None]

    def _run():
        try:
            # pylint: disable=import-outside-toplevel
            from phi.options.engine import run_options_backtest_full

            # Compute entry signal
            entry_signal = None
            if entry_mode == "signal" and ind_names:
                entry_signal = _ow_compute_signal(ohlcv_source, ind_names)

            result_holder[0] = run_options_backtest_full(
                ohlcv=ohlcv_source,
                symbol=ow_sym,
                strategy_type=strategy_type,
                entry_mode=entry_mode,
                entry_signal=entry_signal,
                signal_threshold=sig_threshold,
                dte=dte,
                iv_factor=iv_factor,
                iv_lookback=iv_lookback,
                position_pct=pos_pct,
                otm_pct=otm_pct,
                exit_profit_pct=profit_target,
                exit_stop_pct=stop_loss,
                exit_dte=dte_exit,
                hold_to_expiry=hold_to_expiry,
                max_open_trades=max_trades,
                risk_free_rate=rfr,
                initial_capital=initial_capital,
                periodic_entry_days=periodic_days,
            )
        except Exception as e:  # pylint: disable=broad-except
            exc_holder[0] = e

    th = threading.Thread(target=_run)
    th.start()
    pct = 5
    t0 = time.time()
    while th.is_alive():
        time.sleep(0.25)
        elapsed = time.time() - t0
        pct = min(95, 5 + int(elapsed * 10))
        progress.progress(pct / 100, text=f"Computing options backtest... {pct}%")

    if exc_holder[0]:
        progress.empty()
        st.error(str(exc_holder[0]))
        st.exception(exc_holder[0])
        return

    progress.progress(1.0, text="Complete — 100%")
    time.sleep(0.3)
    progress.empty()

    results = result_holder[0]
    if results:
        _ow_display_results(results, initial_capital)
    else:
        st.error("Options backtest returned no results.")


# ─────────────────────────────────────────────────────────────────────────────
# Run History
# ─────────────────────────────────────────────────────────────────────────────
def render_run_history():
    """Render the history of previous backtest runs."""
    # pylint: disable=import-outside-toplevel
    from phi.run_config import RunHistory
    hist = RunHistory()
    runs = hist.list_runs()
    if not runs:
        st.caption("No runs yet.")
        return
    st.markdown("### Run History")
    for r in runs[:10]:
        with st.expander(r["run_id"]):
            st.json(r.get("results", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Cache Manager
# ─────────────────────────────────────────────────────────────────────────────
def render_cache_manager():
    """Display information about cached datasets."""
    # pylint: disable=import-outside-toplevel
    from phi.data import list_cached_datasets
    datasets = list_cached_datasets()
    st.markdown("### Cache Manager")
    if not datasets:
        st.caption("No cached datasets.")
        return
    st.dataframe(pd.DataFrame(datasets), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Phi-nance Live Workbench",
        page_icon="📊",
        layout="wide"
    )

    st.title("Phi-nance Live Backtest Workbench")
    st.caption(
        "Fetch → Select → Blend → PhiAI → Run — Reproducible. Cached. Dark."
    )

    # ── Fully Automated (one-click) ─────────────────────────────────────────
    with st.expander("⚡ Run Fully Automated", expanded=True):
        st.caption(
            "One click: fetch data → AI picks indicators → tune params → run backtest. "
            "Uses Ollama when available for smarter selection."
        )
        fa_col1, fa_col2, fa_col3 = st.columns(3)
        with fa_col1:
            fa_sym = st.text_input("Symbol", value="SPY", key="fa_sym")
            fa_start = st.date_input("Start", value=date(2020, 1, 1), key="fa_start")
        with fa_col2:
            fa_end = st.date_input("End", value=date(2024, 12, 31), key="fa_end")
            fa_cap = st.number_input("Capital ($)", value=100_000, min_value=1000, key="fa_cap")
        with fa_col3:
            fa_ollama = st.checkbox("Use Ollama for AI selection", value=True, key="fa_ollama")
            fa_host = st.text_input("Ollama host", value="http://localhost:11434", key="fa_host")

        if st.button("Run Fully Automated", type="primary", key="fa_run"):
            _run_fully_automated(
                symbol=fa_sym or "SPY",
                start_date=str(fa_start),
                end_date=str(fa_end),
                capital=fa_cap,
                use_ollama=fa_ollama,
                ollama_host=fa_host,
            )

    st.markdown("---")
    st.markdown("**Or configure step-by-step:**")
    config = render_dataset_builder()
    indicators = render_indicator_selection()
    blend_method = "weighted_sum"
    blend_weights = {}
    if len(indicators) >= 2:
        blend_method, blend_weights = render_blending(indicators)
    render_phiai()
    if config:
        render_backtest_controls(config)

    render_run_and_results(config, indicators, blend_method, blend_weights)

    st.markdown("---")
    tab_opts, tab_hist, tab_cache, tab_agents = st.tabs(
        ["Options Workbench", "Run History", "Cache Manager", "AI Agents"]
    )
    with tab_opts:
        render_options_workbench()
    with tab_hist:
        render_run_history()
    with tab_cache:
        render_cache_manager()
    with tab_agents:
        render_ai_agents()


if __name__ == "__main__":
    main()

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

            progress.progress(0.1, text="Fetching data... 10%")
            cfg, indicators, blend_method, explanation = run_pipeline(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
                ollama_host=ollama_host,
                use_ollama=use_ollama,
            )

            progress.progress(0.5, text="Running backtest... 50%")

            use_blended = len(indicators) >= 2
            if use_blended:
                strat_cls = _load_strategy(
                    "strategies.blended_workbench_strategy.BlendedWorkbenchStrategy"
                )
                params = {
                    "symbol": symbol,
                    "indicators": indicators,
                    "blend_method": blend_method,
                    "blend_weights": {k: 1.0 / len(indicators) for k in indicators},
                    "signal_threshold": 0.15,
                    "lookback_bars": 200,
                }
            else:
                first = list(indicators.keys())[0]
                info = INDICATOR_CATALOG.get(first, {})
                strat_str = info.get("strategy", "strategies.buy_and_hold.BuyAndHold")
                strat_cls = _load_strategy(strat_str)
                p_defaults = {k: default for k, (_, _, default) in info.get("params", {}).items()}
                params = {**p_defaults, **indicators[first].get("params", {})}
                params["symbol"] = symbol

            results, strat = _run_backtest(strat_cls, params, cfg)
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

    if start_d >= end_d:
        st.error("Start date must be before end date.")
        return None

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
        st.error("Enter at least one symbol.")
        return None

    dfs = {}

    date_range_years = (end_d - start_d).days / 365.25
    if date_range_years > 5:
        st.warning(
            "⚠️ Date range > 5 years selected — this may take longer to process. "
            "Consider reducing the date range for faster results."
        )

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
    if len(selected) > 4:
        st.warning(
            "⚠️ More than 4 indicators selected — this may slow down results significantly. "
            "Consider reducing scope for faster performance."
        )
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
        if phiai_full and st.session_state.get("workbench_dataset"):
            phiai_progress = st.progress(0, text="PhiAI optimizing... 0%")
            phiai_result = [None]
            phiai_exc = [None]

            def run_phiai():
                try:
                    dfs = st.session_state["workbench_dataset"]
                    sym = config["symbols"][0]
                    ohlcv = dfs.get(sym)
                    if ohlcv is not None and len(ohlcv) > 100:
                        # pylint: disable=import-outside-toplevel
                        from phi.phiai import run_phiai_optimization
                        phiai_result[0] = run_phiai_optimization(
                            ohlcv, indicators_to_use, max_iter_per_indicator=15
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

        # Equities: use blended or single strategy
        use_blended = len(indicators_to_use) >= 2
        progress = st.progress(0, text="Preparing backtest... 0%")

        try:
            if use_blended:
                strat_path = (
                    "strategies.blended_workbench_strategy"
                    ".BlendedWorkbenchStrategy"
                )
                strat_cls = _load_strategy(strat_path)
                params = {
                    "symbol": config["symbols"][0],
                    "indicators": indicators_to_use,
                    "blend_method": blend_method,
                    "blend_weights": blend_weights,
                    "signal_threshold": 0.15,
                    "lookback_bars": 200,
                }
            else:
                first_name = list(indicators_to_use.keys())[0]
                info = INDICATOR_CATALOG[first_name]
                strat_str = str(info["strategy"])
                strat_cls = _load_strategy(strat_str)
                # pylint: disable=import-outside-toplevel
                p_defaults = {
                    k: default for k, (_, _, default) in info["params"].items()
                }
                p_user = {
                    k: int(v) if isinstance(v, float) and v == int(v) else v
                    for k, v in indicators_to_use[first_name].get(
                        "params", {}
                    ).items()
                }
                params = {**p_defaults, **p_user}
                params["symbol"] = config["symbols"][0]

            # Run backtest in thread with live progress %
            result_holder = [None]
            exc_holder = [None]

            def run_bt():
                try:
                    result_holder[0] = _run_backtest(strat_cls, params, config)
                except Exception as e:  # pylint: disable=broad-except
                    exc_holder[0] = e

            th = threading.Thread(target=run_bt)
            th.start()

            pct = 5
            start_t = time.time()
            while th.is_alive():
                time.sleep(0.4)
                elapsed = time.time() - start_t
                # Estimate progress: ramp 5% -> 95%
                pct = min(95, 5 + int(elapsed * 1.5))
                progress.progress(
                    pct / 100, text=f"Running backtest... {pct}%"
                )

            if (eh := exc_holder[0]) is not None:
                # pylint: disable=raising-bad-type
                raise eh

            if (res_h := result_holder[0]) is not None:
                assert isinstance(res_h, (list, tuple))
                if len(res_h) == 2:
                    # pylint: disable=unpacking-non-sequence
                    results, strat = res_h
                progress.progress(
                    1.0, text="Complete — 100%"
                )
                time.sleep(0.5)
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
    tab_hist, tab_cache, tab_agents = st.tabs(
        ["Run History", "Cache Manager", "AI Agents"]
    )
    with tab_hist:
        render_run_history()
    with tab_cache:
        render_cache_manager()
    with tab_agents:
        render_ai_agents()


if __name__ == "__main__":
    main()

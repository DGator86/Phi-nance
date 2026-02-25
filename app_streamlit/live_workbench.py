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
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("IS_BACKTESTING", "True")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

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
        "description": "Relative Strength Index — UP < oversold; DOWN > overbought.",
        "params": {"rsi_period": (2, 50, 14), "oversold": (10, 50, 30), "overbought": (50, 95, 70)},
        "strategy": "strategies.rsi.RSIStrategy",
    },
    "MACD": {
        "description": "Moving Average Convergence Divergence — bullish/bearish crossover.",
        "params": {"fast_period": (2, 50, 12), "slow_period": (10, 100, 26), "signal_period": (2, 30, 9)},
        "strategy": "strategies.macd.MACDStrategy",
    },
    "Bollinger": {
        "description": "Bollinger Bands — UP below lower band; DOWN above upper.",
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

BLEND_METHODS = ["Weighted Sum", "Regime-Weighted", "Voting", "PhiAI Chooses"]
METRICS = ["ROI", "CAGR", "Sharpe", "Max Drawdown", "Direction Accuracy", "Profit Factor"]
EXIT_STRATEGIES = ["Signal exit", "SL/TP", "Trailing stop", "Time exit"]
POSITION_SIZING = ["Fixed %", "Fixed shares"]

# Visual spec — chart colors (purple/orange theme)
CHART_COLORS = ["#a855f7", "#f97316", "#22c55e", "#06b6d4", "#eab308"]


def _load_strategy(module_cls: str):
    module_path, cls_name = module_cls.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _run_backtest(strategy_class, params: dict, config: dict):
    """
    Run a Lumibot backtest.

    Prefers PandasDataBacktesting (data from local cache — works offline,
    no API rate limits).  Falls back to AlphaVantageFixedDataSource only
    when no cached data is available.
    """
    from lumibot.entities import Asset

    tf = config.get("timeframe", "1D")
    timestep = "day" if tf == "1D" else "minute"
    symbol = config["symbols"][0]
    start = config["start"]
    end = config["end"]
    vendor = config.get("vendor", "yfinance")

    # ── Try to load cached OHLCV data first ──────────────────────────────────
    from phi.data import get_cached_dataset, fetch_and_cache
    df = get_cached_dataset(vendor, symbol, tf, str(start.date()), str(end.date()))

    if df is None or df.empty:
        # Auto-fetch if not cached (yfinance always works, no key needed)
        _api_key = {
            "finnhub":  os.getenv("FINNHUB_API_KEY"),
            "stockdata": os.getenv("STOCKDATA_API_KEY"),
            "massive":  os.getenv("MASSIVE_API_KEY"),
        }.get(vendor, os.getenv("AV_API_KEY"))
        try:
            df = fetch_and_cache(vendor, symbol, tf, str(start.date()), str(end.date()), api_key=_api_key)
        except Exception:
            df = None

    if df is not None and not df.empty:
        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()

        # Build pandas_data dict for PandasDataBacktesting
        asset = Asset(symbol, asset_type=Asset.AssetTypes.STOCK)
        try:
            from lumibot.backtesting import PandasDataBacktesting
            pandas_data = {asset: {"df": df, "timestep": timestep}}
            results, strat = strategy_class.run_backtest(
                datasource_class=PandasDataBacktesting,
                backtesting_start=start,
                backtesting_end=end,
                budget=config["initial_capital"],
                benchmark_asset=config.get("benchmark", symbol),
                parameters=params,
                pandas_data=pandas_data,
                show_plot=False,
                show_tearsheet=False,
                save_tearsheet=False,
                show_indicators=False,
                show_progress_bar=False,
                quiet_logs=True,
            )
            return results, strat
        except Exception as e:
            # PandasDataBacktesting failed — log and fall through to AV
            print(f"PandasDataBacktesting failed ({e}), falling back to AV data source")

    # ── Fallback: AlphaVantageFixedDataSource ────────────────────────────────
    from strategies.alpha_vantage_fixed import AlphaVantageFixedDataSource
    av_api_key = os.getenv("AV_API_KEY", "PLN25H3ESMM1IRBN")
    results, strat = strategy_class.run_backtest(
        datasource_class=AlphaVantageFixedDataSource,
        backtesting_start=start,
        backtesting_end=end,
        budget=config["initial_capital"],
        benchmark_asset=config.get("benchmark", symbol),
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
    from strategies.prediction_tracker import compute_prediction_accuracy
    return compute_prediction_accuracy(strat)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────
def render_dataset_builder():
    st.markdown("### Step 1 — Dataset Builder")

    col_mode, col_sym, col_range = st.columns([1, 2, 2])
    with col_mode:
        trading_mode = st.selectbox("Trading Mode", ["Equities", "Options"], key="ds_mode")
    with col_sym:
        symbols_raw = st.text_input("Symbol(s)", value="SPY", key="ds_symbols",
                                    help="Comma-separated: SPY, QQQ, AAPL")
    with col_range:
        start_d = st.date_input("Start", value=date(2020, 1, 1), key="ds_start")
        end_d = st.date_input("End", value=date(2024, 12, 31), key="ds_end")

    col_tf, col_cap = st.columns(2)
    with col_tf:
        timeframe = st.selectbox("Timeframe", ["1D", "4H", "1H", "15m", "5m", "1m"], key="ds_tf")
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
        fetch_clicked = st.button("Fetch & Cache Data", type="primary", key="ds_fetch")
    with col_use:
        use_cached = st.button("Use Cached Data", key="ds_use")

    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        st.warning("Enter at least one symbol.")
        return None

    dataset_ready = False
    dfs = {}

    if fetch_clicked or use_cached:
        from phi.data import auto_fetch_and_cache, get_cached_dataset

        with st.status("Loading data...", expanded=True) as s:
            for sym in symbols:
                try:
                    if fetch_clicked:
                        df, vendor_used = auto_fetch_and_cache(
                            sym, timeframe, str(start_d), str(end_d)
                        )
                        s.write(f"{sym}: fetched {len(df):,} bars via {vendor_used}")
                    else:
                        # Try each vendor cache in priority order
                        df = None
                        for _v in ("massive", "finnhub", "yfinance", "alphavantage", "stockdata"):
                            df = get_cached_dataset(_v, sym, timeframe, str(start_d), str(end_d))
                            if df is not None and not df.empty:
                                break
                        if df is None or df.empty:
                            raise ValueError("No cached data — click 'Fetch & Cache Data'")
                    if df is not None and not df.empty:
                        dfs[sym] = df
                except Exception as e:
                    st.error(f"{sym}: {e}")
            if dfs:
                st.session_state["workbench_dataset"] = dfs
                st.session_state["workbench_config"] = {
                    "trading_mode": trading_mode.lower(),
                    "symbols": symbols,
                    "start": datetime.combine(start_d, datetime.min.time()),
                    "end": datetime.combine(end_d, datetime.min.time()),
                    "timeframe": timeframe,
                    "vendor": "auto",
                    "initial_capital": float(initial_capital),
                    "benchmark": symbols[0],
                }
                s.update(label=f"Ready — {sum(len(d) for d in dfs.values()):,} bars", state="complete")
                dataset_ready = True
            else:
                s.update(label="No data loaded", state="error")

    if st.session_state.get("workbench_dataset"):
        dfs = st.session_state["workbench_dataset"]
        cfg = st.session_state.get("workbench_config", {})
        st.success(f"**Dataset ready:** {', '.join(dfs.keys())} · {sum(len(d) for d in dfs.values()):,} bars · "
                   f"${cfg.get('initial_capital', 0):,.0f} initial capital")
        for sym, df in list(dfs.items())[:3]:
            with st.expander(f"{sym} — {len(df):,} rows"):
                st.dataframe(df.tail(20), use_container_width=True)
        return cfg
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Indicator Selection
# ─────────────────────────────────────────────────────────────────────────────
def render_indicator_selection():
    st.markdown("### Step 2 — Indicator Selection")

    selected = st.session_state.get("workbench_indicators", {})

    left, right = st.columns([1, 1])
    with left:
        search = st.text_input("Search indicators", key="ind_search", placeholder="RSI, MACD...")
        available = [k for k in INDICATOR_CATALOG if not search or search.lower() in k.lower()]
        for name in available:
            info = INDICATOR_CATALOG[name]
            enabled = st.checkbox(f"**{name}** — {info['description'][:50]}...", value=name in selected, key=f"ind_{name}")
            if enabled:
                if name not in selected:
                    selected[name] = {"enabled": True, "auto_tune": False, "params": {}}
                selected[name]["enabled"] = True
                selected[name]["auto_tune"] = st.toggle("PhiAI Auto-tune", value=selected[name].get("auto_tune", False), key=f"at_{name}")
                with st.expander("Manual tuning", expanded=False):
                    for pname, (lo, hi, default) in info["params"].items():
                        selected[name]["params"][pname] = st.slider(pname, lo, hi, default, key=f"param_{name}_{pname}")
            else:
                if name in selected:
                    del selected[name]

    st.session_state["workbench_indicators"] = selected
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Blending Panel
# ─────────────────────────────────────────────────────────────────────────────
def render_blending(indicators: dict):
    if len(indicators) < 2:
        st.caption("Select 2+ indicators to enable blending.")
        return "weighted_sum", {}

    st.markdown("### Step 3 — Blending Panel")
    method = st.selectbox("Blend Mode", BLEND_METHODS, key="blend_method")
    method_map = {"Weighted Sum": "weighted_sum", "Regime-Weighted": "regime_weighted",
                  "Voting": "voting", "PhiAI Chooses": "phiai_chooses"}
    method_key = method_map.get(method, "weighted_sum")

    weights = {}
    for name in indicators:
        weights[name] = st.slider(f"Weight: {name}", 0.0, 1.0, 1.0 / len(indicators), 0.05, key=f"wt_{name}")
    return method_key, weights


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — PhiAI Panel
# ─────────────────────────────────────────────────────────────────────────────
def render_phiai():
    st.markdown("### Step 4 — PhiAI Panel")
    phiai_full = st.toggle("PhiAI Full Auto", value=False, key="phiai_full",
                           help="Auto-enable/disable indicators, tune params, select blend")
    if phiai_full:
        st.info("PhiAI will optimize indicators, parameters, and blend. Regime-aware adjustments applied.")
        max_ind = st.number_input("Max indicators", 1, 10, 5, key="phiai_max")
        no_shorts = st.checkbox("No shorting", value=True, key="phiai_noshort")
    return phiai_full


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Backtest Controls
# ─────────────────────────────────────────────────────────────────────────────
def render_backtest_controls(config: dict):
    if not config:
        return {}

    st.markdown("### Step 5 — Backtest Controls")
    mode = config.get("trading_mode", "equities")

    if mode == "equities":
        allow_short = st.checkbox("Allow shorting", value=False, key="bt_short")
        pos_sizing = st.selectbox("Position sizing", POSITION_SIZING, key="bt_pos")
        exit_strat = st.selectbox("Exit strategy", EXIT_STRATEGIES, key="bt_exit")
        return {"allow_short": allow_short, "position_sizing": pos_sizing, "exit_strategy": exit_strat}
    else:
        st.caption("Options mode: Long Call/Put with delta-based simulation.")
        strat_type = st.selectbox("Strategy", ["long_call", "long_put"], key="opt_strat")
        exit_profit = st.slider("Exit profit %", 0.2, 1.0, 0.5, 0.1, key="opt_exit_profit")
        exit_stop = st.slider("Exit stop %", -0.5, -0.1, -0.3, 0.05, key="opt_exit_stop")
        opts = {"strategy_type": strat_type, "exit_profit_pct": exit_profit, "exit_stop_pct": exit_stop}
        st.session_state["bt_options_controls"] = opts
        return opts


# ─────────────────────────────────────────────────────────────────────────────
# Run & Results
# ─────────────────────────────────────────────────────────────────────────────
def render_run_and_results(config: dict, indicators: dict, blend_method: str, blend_weights: dict):
    if not config or not indicators:
        st.info("Complete Steps 1–2 to run a backtest.")
        return

    st.markdown("---")
    st.markdown("## Run Backtest")

    primary_metric = st.selectbox("Primary metric", METRICS, key="primary_metric")
    col_run, col_stop = st.columns(2)
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
                        from phi.phiai import run_phiai_optimization
                        phiai_result[0] = run_phiai_optimization(
                            ohlcv, indicators_to_use, max_iter_per_indicator=15
                        )
                except Exception as ex:
                    phiai_exc[0] = ex

            th_phiai = threading.Thread(target=run_phiai)
            th_phiai.start()
            pct = 0
            start_t = time.time()
            while th_phiai.is_alive():
                time.sleep(0.3)
                elapsed = time.time() - start_t
                pct = min(95, int(elapsed * 12))  # ~8 sec to 95%
                phiai_progress.progress(pct / 100, text=f"PhiAI optimizing... {pct}%")
            if phiai_exc[0]:
                st.warning(f"PhiAI optimization skipped: {phiai_exc[0]}")
            elif phiai_result[0]:
                indicators_to_use, phiai_explanation = phiai_result[0]
                st.session_state["phiai_explanation"] = phiai_explanation
            phiai_progress.progress(1.0, text="PhiAI complete — 100%")
            time.sleep(0.3)
            phiai_progress.empty()

        # Options mode: use phi.options.backtest
        if trading_mode == "options":
            try:
                opt_progress = st.progress(0, text="Running options backtest... 0%")
                opt_result = [None]
                opt_exc = [None]

                def run_opt():
                    try:
                        dfs = st.session_state.get("workbench_dataset", {})
                        sym = config["symbols"][0]
                        ohlcv = dfs.get(sym)
                        if ohlcv is None:
                            from phi.data import get_cached_dataset
                            ohlcv = get_cached_dataset("alphavantage", sym, "1D",
                                                       str(config["start"].date()), str(config["end"].date()))
                        if ohlcv is None:
                            from phi.data import fetch_and_cache
                            ohlcv = fetch_and_cache("alphavantage", sym, "1D",
                                                    str(config["start"].date()), str(config["end"].date()))
                        if ohlcv is None or ohlcv.empty:
                            raise ValueError("No data for options backtest. Fetch data first.")
                        bt_opts = st.session_state.get("bt_options_controls", {})
                        from phi.options import run_options_backtest
                        opt_result[0] = run_options_backtest(
                            ohlcv,
                            strategy_type=bt_opts.get("strategy_type", "long_call"),
                            initial_capital=config.get("initial_capital", 100_000),
                            position_pct=0.1,
                            exit_profit_pct=bt_opts.get("exit_profit_pct", 0.5),
                            exit_stop_pct=bt_opts.get("exit_stop_pct", -0.3),
                        )
                    except Exception as e:
                        opt_exc[0] = e

                th_opt = threading.Thread(target=run_opt)
                th_opt.start()
                pct = 0
                start_t = time.time()
                while th_opt.is_alive():
                    time.sleep(0.2)
                    elapsed = time.time() - start_t
                    pct = min(95, int(elapsed * 25))  # ramps quickly
                    opt_progress.progress(pct / 100, text=f"Running options backtest... {pct}%")
                if opt_exc[0]:
                    opt_progress.empty()
                    raise opt_exc[0]
                results = opt_result[0]
                opt_progress.progress(1.0, text="Complete — 100%")
                time.sleep(0.3)
                opt_progress.empty()
                _display_results(config, results, None, indicators_to_use, blend_method, blend_weights)
            except Exception as e:
                st.error(str(e))
                st.exception(e)
            return

        # Equities: use blended or single strategy
        use_blended = len(indicators_to_use) >= 2
        progress = st.progress(0, text="Preparing backtest... 0%")

        try:
            if use_blended:
                strat_cls = _load_strategy("strategies.blended_workbench_strategy.BlendedWorkbenchStrategy")
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
                strat_cls = _load_strategy(info["strategy"])
                params = {**{k: default for k, (_, _, default) in info["params"].items()},
                          **{k: int(v) if isinstance(v, float) and v == int(v) else v
                             for k, v in indicators_to_use[first_name].get("params", {}).items()}}
                params["symbol"] = config["symbols"][0]

            # Run backtest in thread with live progress %
            result_holder = [None]
            exc_holder = [None]

            def run_bt():
                try:
                    result_holder[0] = _run_backtest(strat_cls, params, config)
                except Exception as e:
                    exc_holder[0] = e

            th = threading.Thread(target=run_bt)
            th.start()

            pct = 5
            start_t = time.time()
            while th.is_alive():
                time.sleep(0.4)
                elapsed = time.time() - start_t
                # Estimate progress: ramp 5% -> 95% over ~60 seconds
                pct = min(95, 5 + int(elapsed * 1.5))
                progress.progress(pct / 100, text=f"Running backtest... {pct}%")

            if exc_holder[0]:
                raise exc_holder[0]

            results, strat = result_holder[0]
            progress.progress(1.0, text="Complete — 100%")
            time.sleep(0.5)
            progress.empty()

            sc = _compute_accuracy(strat) if hasattr(strat, "prediction_log") else {}
            _display_results(config, results, strat, indicators_to_use, blend_method, blend_weights, sc)

            if phiai_full and st.session_state.get("phiai_explanation"):
                with st.expander("PhiAI changes"):
                    st.text(st.session_state["phiai_explanation"])

        except Exception as e:
            progress.empty()
            st.error(str(e))
            st.exception(e)


def _extract_scalar(val):
    """Extract a scalar float from a value that may be a dict (lumibot wraps some metrics)."""
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


def _display_results(config, results, strat, indicators, blend_method, blend_weights, sc=None):
    sc = sc or {}
    tr = _extract_scalar(getattr(results, "total_return", None) or results.get("total_return"))
    cagr = _extract_scalar(getattr(results, "cagr", None) or results.get("cagr"))
    dd = _extract_scalar(getattr(results, "max_drawdown", None) or results.get("max_drawdown"))
    sharpe = _extract_scalar(getattr(results, "sharpe", None) or results.get("sharpe"))
    cap = config.get("initial_capital", 100_000)
    pv = getattr(results, "portfolio_value", None) or results.get("portfolio_value")
    end_cap = pv[-1] if pv and len(pv) else cap
    net_pl = end_cap - cap
    net_pct = (net_pl / cap) * 100 if cap else 0

    st.markdown("### Results")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Start Capital", f"${cap:,.0f}")
    r2.metric("End Capital", f"${end_cap:,.0f}")
    r3.metric("Net P/L", f"${net_pl:+,.0f}", f"{net_pct:+.1f}%")
    r4.metric("CAGR", f"{cagr:+.1%}" if cagr is not None else "—")
    r5.metric("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "—")

    tab_sum, tab_curve, tab_trades, tab_metrics = st.tabs(["Summary", "Equity Curve", "Trades", "Metrics"])
    with tab_sum:
        st.metric("Max Drawdown", f"{dd:.1%}" if dd is not None else "—")
        st.metric("Direction Accuracy", f"{sc.get('accuracy', 0):.1%}" if sc else "—")
    with tab_curve:
        if pv and len(pv) > 1:
            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=pv, mode="lines", line=dict(color=CHART_COLORS[0], width=2)))
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0f0f12",
                    plot_bgcolor="#1a1a1f",
                    font_color="#e4e4e7",
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(pd.DataFrame(pv, columns=["Portfolio Value ($)"]))
    with tab_trades:
        if strat and hasattr(strat, "prediction_log") and strat.prediction_log:
            st.dataframe(pd.DataFrame(strat.prediction_log), use_container_width=True)
        else:
            st.caption("No trade log.")
    with tab_metrics:
        st.json({"total_return": tr, "cagr": cagr, "max_drawdown": dd, "sharpe": sharpe, "accuracy": sc.get("accuracy")})

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
    hist.save_results(run_id, {"total_return": tr, "cagr": cagr, "max_drawdown": dd, "sharpe": sharpe, "accuracy": sc.get("accuracy"), "net_pl": net_pl})
    st.caption(f"Run saved: {run_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Run History
# ─────────────────────────────────────────────────────────────────────────────
def render_run_history():
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
    from phi.data import list_cached_datasets
    datasets = list_cached_datasets()
    st.markdown("### Cache Manager")
    if not datasets:
        st.caption("No cached datasets.")
        return
    st.dataframe(pd.DataFrame(datasets), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# AI Backtest Agent UI
# ─────────────────────────────────────────────────────────────────────────────
def render_ai_agent():
    """
    AI Backtest Agent tab.

    Runs every strategy with multiple parameter sets, ranks results, sends
    them to Claude for analysis, and writes the winning configuration to
    data_cache/learned_params/{SYMBOL}_{TF}.json so the active trading
    agent can load it on startup.
    """
    st.markdown("## AI Backtest Agent")
    st.caption(
        "Set a symbol and date range — the agent tests every built-in strategy, "
        "ranks them, and asks Claude to distill the lessons into a live trading config."
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        symbol = st.text_input("Symbol", value="SPY", key="ai_symbol").strip().upper()
    with col2:
        start_d = st.date_input("Start", value=date(2022, 1, 1), key="ai_start")
    with col3:
        end_d = st.date_input("End", value=date(2024, 12, 31), key="ai_end")
    with col4:
        capital = st.number_input(
            "Capital ($)", value=100_000, min_value=1_000, step=10_000, key="ai_cap"
        )

    # Show existing learned params for this symbol if any
    from phi.agents import BacktestAgent
    agent = BacktestAgent()
    existing = agent.load_learned_params(symbol)
    if existing:
        with st.expander(
            f"Learned params on file for {symbol} "
            f"(last run: {existing.get('updated_at', '?')[:10]})",
            expanded=False,
        ):
            best = existing.get("best_run", {})
            perf = best.get("performance", {})
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Best Strategy", best.get("name", "—"))
            col_b.metric("Sharpe", f"{perf.get('sharpe', 0):.2f}")
            col_c.metric("CAGR", f"{perf.get('cagr', 0)*100:.1f}%")
            col_d.metric("Max DD", f"{perf.get('max_drawdown', 0)*100:.1f}%")
            if existing.get("ai_analysis_summary"):
                st.markdown(existing["ai_analysis_summary"][:400] + "…")

    run_clicked = st.button("Run AI Backtest Agent", type="primary", key="ai_run")
    if not run_clicked:
        return

    if not symbol:
        st.warning("Enter a symbol first.")
        return
    if start_d >= end_d:
        st.error("Start must be before End.")
        return

    # ── Run the agent with live Streamlit progress ────────────────────────────
    results_store: dict = {}
    status_slots: dict = {}         # label → st.status handle
    metric_slots: dict = {}         # label → st.empty handle
    progress_bar = st.progress(0.0)
    total_runs = sum(
        1 + (1 if name != "Buy & Hold" else 0)
        for name in ["RSI", "MACD", "Bollinger", "Dual SMA",
                     "Mean Reversion", "Breakout", "Buy & Hold"]
    ) + 1  # +1 for Claude

    completed = [0]  # mutable counter for closure

    def on_progress(label: str, status: str, metrics: dict | None = None):
        if status == "fetching":
            status_slots[label] = st.status(f"Fetching {symbol} market data…", expanded=False)
        elif status == "complete" and label == "Data":
            v = (metrics or {}).get("vendor", "auto")
            b = (metrics or {}).get("bars", 0)
            status_slots.get("Data", st.empty()).update(
                label=f"Data ready — {b:,} bars via {v}", state="complete"
            )
        elif status == "running":
            status_slots[label] = st.status(f"Testing {label}…", expanded=False)
            metric_slots[label] = st.empty()
        elif status == "complete" and label != "Data":
            m = metrics or {}
            if label == "Claude Analysis":
                status_slots.get(label, st.empty()).update(
                    label="Claude analysis complete", state="complete"
                )
            elif m.get("status") == "ok":
                status_slots.get(label, st.empty()).update(
                    label=(
                        f"{label} — "
                        f"Sharpe {m.get('sharpe', 0):.2f}  |  "
                        f"CAGR {m.get('cagr', 0)*100:.1f}%  |  "
                        f"MaxDD {m.get('max_drawdown', 0)*100:.1f}%"
                    ),
                    state="complete",
                )
            else:
                status_slots.get(label, st.empty()).update(
                    label=f"{label} — error", state="error"
                )
            completed[0] += 1
            progress_bar.progress(min(completed[0] / total_runs, 1.0))
        elif status == "error":
            status_slots.get(label, st.empty()).update(
                label=f"{label} — failed", state="error"
            )
            completed[0] += 1
            progress_bar.progress(min(completed[0] / total_runs, 1.0))

    try:
        result = agent.run(
            symbol=symbol,
            start=str(start_d),
            end=str(end_d),
            capital=float(capital),
            timeframe="1D",
            on_progress=on_progress,
        )
        results_store.update(result)
    except Exception as exc:
        st.error(f"Agent failed: {exc}")
        return

    progress_bar.progress(1.0)

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Agent Results")

    best = results_store.get("best")
    if best:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Strategy", best["name"])
        c2.metric("Sharpe Ratio", f"{best.get('sharpe', 0):.2f}")
        c3.metric("CAGR", f"{best.get('cagr', 0)*100:.1f}%")
        c4.metric("Max Drawdown", f"{best.get('max_drawdown', 0)*100:.1f}%")

    # All runs table
    ok_runs = [r for r in results_store.get("runs", []) if r["status"] == "ok"]
    if ok_runs:
        ok_runs_sorted = sorted(ok_runs, key=lambda r: r.get("score", 0), reverse=True)
        df_runs = pd.DataFrame([
            {
                "Strategy": r["name"],
                "Sharpe":   round(r.get("sharpe", 0), 2),
                "CAGR %":   round(r.get("cagr", 0) * 100, 1),
                "Max DD %": round(r.get("max_drawdown", 0) * 100, 1),
                "Return %": round(r.get("total_return", 0) * 100, 1),
                "Score":    round(r.get("score", 0), 2),
            }
            for r in ok_runs_sorted
        ])
        st.dataframe(df_runs, use_container_width=True, hide_index=True)

    # AI analysis
    ai_text = results_store.get("ai_analysis", "")
    if ai_text:
        with st.expander("Claude's Analysis & Lessons", expanded=True):
            st.markdown(ai_text)

    # Learned params saved confirmation
    lp = results_store.get("learned_params_path", "")
    if lp:
        st.success(
            f"Learned parameters saved to `{lp}`. "
            "The active trading agent will load these on its next run."
        )
        ai_params = results_store.get("ai_params", {})
        if ai_params:
            with st.expander("Recommended live trading config (from Claude)"):
                st.json(ai_params)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Phi-nance Live Workbench", page_icon="📊", layout="wide")

    st.title("Phi-nance Live Backtest Workbench")

    tab_agent, tab_manual, tab_hist, tab_cache = st.tabs([
        "AI Backtest Agent",
        "Manual Workbench",
        "Run History",
        "Cache",
    ])

    with tab_agent:
        render_ai_agent()

    with tab_manual:
        config = render_dataset_builder()
        indicators = render_indicator_selection()
        blend_method = "weighted_sum"
        blend_weights = {}
        if len(indicators) >= 2:
            blend_method, blend_weights = render_blending(indicators)
        render_phiai()
        render_backtest_controls(config) if config else {}
        render_run_and_results(config, indicators, blend_method, blend_weights)

    with tab_hist:
        render_run_history()

    with tab_cache:
        render_cache_manager()


if __name__ == "__main__":
    main()

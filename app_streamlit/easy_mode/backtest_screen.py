"""One-click regime-weighted backtest: multiple entry strictness presets, best curve highlighted."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from phi.backtest.direct import run_direct_backtest
from phi.regime.train import train_regime_detector

from app_streamlit.easy_mode.constants import LOOKBACK_DAYS, PRIMARY_BACKTEST_SYMBOL
from app_streamlit.easy_mode.data import load_ohlcv

# Curated stack — same for every run; only entry strictness (signal threshold) varies.
_AUTO_INDICATORS: dict[str, dict[str, Any]] = {
    "RSI": {"enabled": True, "params": {"rsi_period": 14, "oversold": 30, "overbought": 70}},
    "MACD": {"enabled": True, "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}},
    "Bollinger": {"enabled": True, "params": {"bb_period": 20, "num_std": 2}},
    "Dual SMA": {"enabled": True, "params": {"fast_period": 10, "slow_period": 50}},
    "Buy & Hold": {"enabled": True, "params": {}},
}
_n_ind = len(_AUTO_INDICATORS)
_BASE_WEIGHTS = {k: round(1.0 / _n_ind, 4) for k in _AUTO_INDICATORS}
# Slight drift fix
_first = next(iter(_BASE_WEIGHTS))
_BASE_WEIGHTS[_first] = round(_BASE_WEIGHTS[_first] + (1.0 - sum(_BASE_WEIGHTS.values())), 4)

_PRESETS: tuple[tuple[str, float], ...] = (
    ("Cautious (fewer trades)", 0.22),
    ("Balanced", 0.16),
    ("Responsive (more trades)", 0.10),
)


def render_auto_backtest() -> None:
    st.markdown('<p class="phinance-hero">Automatic backtest</p>', unsafe_allow_html=True)
    st.caption(
        f"Regime detection runs in the background (3 clusters on {PRIMARY_BACKTEST_SYMBOL}). "
        "Signals are blended **regime-weighted** with the stack below — you only choose when to run."
    )
    with st.expander("What’s under the hood (read-only)", expanded=False):
        st.write(
            "**Indicators:** RSI, MACD, Bollinger, Dual SMA, Buy & Hold — equal base weights; "
            "per-bar regime adjusts effective weights via the engine defaults."
        )
        st.write("**Entry / exit:** long when composite signal clears the threshold; flat when it fades.")
        st.json({"presets": [{"name": n, "signal_threshold": t} for n, t in _PRESETS]})

    if st.button("Run automatic analysis", type="primary", use_container_width=True):
        with st.spinner("Fitting regime model and running presets…"):
            try:
                ohlcv = load_ohlcv(PRIMARY_BACKTEST_SYMBOL, LOOKBACK_DAYS)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return

            detector, _ = train_regime_detector(
                ohlcv,
                method="kmeans",
                n_regimes=3,
                window=20,
                save=False,
            )
            regime_series = detector.predict(ohlcv)

            results_by_name: dict[str, dict[str, Any]] = {}
            for label, thresh in _PRESETS:
                res, _ = run_direct_backtest(
                    ohlcv=ohlcv,
                    symbol=PRIMARY_BACKTEST_SYMBOL,
                    indicators=_AUTO_INDICATORS,
                    blend_weights=dict(_BASE_WEIGHTS),
                    blend_method="regime_weighted",
                    signal_threshold=float(thresh),
                    initial_capital=100_000.0,
                    position_size_pct=0.95,
                    regime_series=regime_series,
                    regime_label_map=None,
                    regime_boosts={},
                )
                results_by_name[label] = res

            best_name = max(
                results_by_name,
                key=lambda k: float(results_by_name[k].get("sharpe", -999) or -999),
            )

            st.session_state["easy_last_backtest"] = {
                "symbol": PRIMARY_BACKTEST_SYMBOL,
                "best": best_name,
                "results": results_by_name,
                "regime_series": regime_series,
                "ohlcv": ohlcv,
            }

    state = st.session_state.get("easy_last_backtest")
    if not state:
        st.info("Tap **Run automatic analysis** to compare three entry styles on the same history.")
        return

    results_by_name: dict[str, dict[str, Any]] = state["results"]
    best_name = state["best"]

    st.success(f"**Best fit this run:** {best_name} (highest Sharpe among presets).")

    cols = st.columns(len(_PRESETS))
    for i, (label, _) in enumerate(_PRESETS):
        r = results_by_name[label]
        cols[i].metric(
            label.split("(")[0].strip(),
            f"{float(r.get('total_return', 0)) * 100:.1f}%",
            help=f"Sharpe {float(r.get('sharpe', 0)):.2f} · max DD {float(r.get('max_drawdown', 0)) * 100:.1f}%",
        )

    fig = go.Figure()
    colors = ("#64748b", "#38bdf8", "#a78bfa")
    for idx, (label, _) in enumerate(_PRESETS):
        pv = results_by_name[label].get("portfolio_value") or []
        if not pv:
            continue
        line = dict(width=4 if label == best_name else 1.8, dash="solid" if label == best_name else "dot")
        fig.add_trace(
            go.Scatter(
                y=pv,
                name=f"{label} ★" if label == best_name else label,
                mode="lines",
                line={**line, "color": colors[idx % len(colors)]},
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15,23,42,0.6)",
        plot_bgcolor="rgba(15,23,42,0.3)",
        title=f"Portfolio value — {state['symbol']} (★ = best Sharpe)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full metrics table"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Preset": k,
                        "Total return %": round(float(v.get("total_return", 0)) * 100, 2),
                        "CAGR %": round(float(v.get("cagr", 0)) * 100, 2),
                        "Sharpe": round(float(v.get("sharpe", 0)), 3),
                        "Max DD %": round(float(v.get("max_drawdown", 0)) * 100, 2),
                    }
                    for k, v in results_by_name.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

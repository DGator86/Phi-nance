"""One-click regime-weighted backtest: full catalog, semantic cluster boosts, API-compat shim."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_streamlit.config import INDICATOR_SPECS
from phi.blending.blender import DEFAULT_REGIME_BOOSTS
from phi.regime.train import train_regime_detector

from app_streamlit.easy_mode.backtest_core import (
    PRESETS,
    build_auto_indicators_and_weights,
    call_run_direct_backtest,
    semantic_label_map_for_clusters,
)
from app_streamlit.easy_mode.backtest_extras import (
    bootstrap_sharpe_distribution,
    run_rsi_threshold_heatmap,
    slice_replay_window,
    train_multi_window_regimes,
)
from app_streamlit.easy_mode.constants import LOOKBACK_DAYS, PRIMARY_BACKTEST_SYMBOL
from app_streamlit.easy_mode.data import load_ohlcv


def _thrust_style(ret: float) -> tuple[str, str]:
    """Return (label, soft background hex) for horizon return."""
    r = ret * 100
    if r > 2.0:
        return f"+{r:.1f}%", "#14532d"
    if r > 0.5:
        return f"+{r:.1f}%", "#166534"
    if r < -2.0:
        return f"{r:.1f}%", "#7f1d1d"
    if r < -0.5:
        return f"{r:.1f}%", "#991b1b"
    return f"{r:+.1f}%", "#334155"


def _mapped_regime_series(
    ohlcv: pd.DataFrame,
    regime_series: pd.Series,
    regime_label_map: dict[str, str],
) -> pd.Series:
    raw = regime_series.reindex(ohlcv.index).ffill().astype(str)
    if not regime_label_map:
        return raw
    return raw.map(lambda x: regime_label_map.get(x, x))


def _build_visual_replay_figure(
    ohlcv: pd.DataFrame,
    best: dict[str, Any],
    mapped_regimes: pd.Series,
    title: str,
) -> go.Figure:
    """Price + equity (secondary Y), regime shading, buy/sell markers with trigger hover."""
    idx = ohlcv.index
    close = ohlcv["close"].astype(float)
    pv = best.get("portfolio_value") or []
    if len(pv) >= len(idx) + 1:
        eq_x = idx
        eq_y = list(pv[1 : 1 + len(idx)])
    elif len(pv) == len(idx):
        eq_x, eq_y = idx, list(pv)
    else:
        eq_y = list(pv[1:] if len(pv) > 1 else pv)
        eq_x = idx[-len(eq_y) :] if len(eq_y) <= len(idx) else idx

    n = min(len(eq_x), len(eq_y))
    if n <= 0:
        eq_x, eq_y = idx[:0], []
    else:
        eq_x = eq_x[-n:]
        eq_y = eq_y[-n:]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    mr = mapped_regimes.reindex(idx).ffill()
    color_map = {
        "TREND_UP": "rgba(34,197,94,0.14)",
        "TREND_DN": "rgba(239,68,68,0.14)",
        "RANGE": "rgba(148,163,184,0.18)",
    }
    changes = mr.ne(mr.shift())
    gid = changes.cumsum()
    for _, block in mr.groupby(gid):
        lab = str(block.iloc[0])
        if lab in ("nan", "None") or pd.isna(block.iloc[0]):
            continue
        c = color_map.get(lab, "rgba(100,116,139,0.12)")
        fig.add_vrect(
            x0=block.index[0],
            x1=block.index[-1],
            fillcolor=c,
            layer="below",
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=idx,
            y=close,
            name="Close",
            line=dict(color="#38bdf8", width=2),
        ),
        secondary_y=False,
    )

    trades = best.get("trade_events") or []
    buys = [t for t in trades if t.get("side") == "buy"]
    sells = [t for t in trades if t.get("side") == "sell"]
    if buys:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in buys],
                y=[t["price"] for t in buys],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=11, color="#4ade80", line=dict(width=0)),
                customdata=[[t.get("trigger", ""), t.get("shares", "")] for t in buys],
                hovertemplate="Buy<br>%{x}<br>px %{y:.2f}<br>%{customdata[0]}<br>sh %{customdata[1]}<extra></extra>",
            ),
            secondary_y=False,
        )
    if sells:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in sells],
                y=[t["price"] for t in sells],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=11, color="#f87171", line=dict(width=0)),
                customdata=[[t.get("trigger", ""), t.get("shares", "")] for t in sells],
                hovertemplate="Sell<br>%{x}<br>px %{y:.2f}<br>%{customdata[0]}<br>sh %{customdata[1]}<extra></extra>",
            ),
            secondary_y=False,
        )

    if eq_y:
        fig.add_trace(
            go.Scatter(
                x=eq_x,
                y=eq_y,
                name="Equity ($)",
                line=dict(color="#c4b5fd", width=2),
            ),
            secondary_y=True,
        )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=520,
        paper_bgcolor="rgba(15,23,42,0.6)",
        plot_bgcolor="rgba(15,23,42,0.25)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=70, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="Equity", secondary_y=True, showgrid=False)
    return fig


def render_auto_backtest() -> None:
    st.markdown('<p class="phinance-hero">Automatic backtest</p>', unsafe_allow_html=True)
    st.caption(
        f"Regime detection (3 clusters on {PRIMARY_BACKTEST_SYMBOL}) plus the **full workbench signal stack** "
        "(core, order-flow, information theory, MFT). Clusters are ranked by average bar return and mapped to "
        "**TREND_DN / RANGE / TREND_UP** so the same per-regime boost table as the expert UI applies."
    )
    auto_inds, _ = build_auto_indicators_and_weights()
    n_ind = len(auto_inds)
    cats: dict[str, int] = {}
    for name in auto_inds:
        cats[INDICATOR_SPECS[name].category] = cats.get(INDICATOR_SPECS[name].category, 0) + 1
    with st.expander("What’s under the hood (read-only)", expanded=False):
        st.write(
            f"**Indicators ({n_ind}):** all names in `INDICATOR_SPECS` that have a computer — "
            "including MFT Signal / MFT Energy, order-flow proxies, entropy / MI / Fisher / KL, plus core oscillators."
        )
        st.caption("By category: " + " · ".join(f"{k} ({v})" for k, v in sorted(cats.items())))
        st.write(
            "**Regime boosts:** `DEFAULT_REGIME_BOOSTS` from the blending engine (e.g. MACD emphasis in trends, "
            "RSI/Bollinger in range) after cluster→semantic mapping."
        )
        st.write("**Entry / exit:** long when composite signal clears the threshold; flat when it fades.")
        st.json({"presets": [{"name": n, "signal_threshold": t} for n, t in PRESETS]})

    if st.button("Run automatic analysis", type="primary", use_container_width=True):
        with st.spinner("Fitting regime model and running presets…"):
            try:
                ohlcv = load_ohlcv(PRIMARY_BACKTEST_SYMBOL, LOOKBACK_DAYS)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return

            indicators, base_weights = build_auto_indicators_and_weights()

            detector, _ = train_regime_detector(
                ohlcv,
                method="kmeans",
                n_regimes=3,
                window=20,
                save=False,
            )
            regime_series = detector.predict(ohlcv)
            regime_label_map = semantic_label_map_for_clusters(ohlcv, regime_series)
            regime_boosts = deepcopy(DEFAULT_REGIME_BOOSTS)
            multi_regime = train_multi_window_regimes(ohlcv)

            results_by_name: dict[str, dict[str, Any]] = {}
            for label, thresh in PRESETS:
                res, _ = call_run_direct_backtest(
                    ohlcv=ohlcv,
                    symbol=PRIMARY_BACKTEST_SYMBOL,
                    indicators=indicators,
                    blend_weights=dict(base_weights),
                    blend_method="regime_weighted",
                    signal_threshold=float(thresh),
                    initial_capital=100_000.0,
                    position_size_pct=0.95,
                    regime_series=regime_series,
                    regime_label_map=regime_label_map or None,
                    regime_boosts=regime_boosts,
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
                "regime_label_map": regime_label_map,
                "n_indicators": len(indicators),
                "multi_regime": multi_regime,
            }

    state = st.session_state.get("easy_last_backtest")
    if not state:
        st.info("Tap **Run automatic analysis** to compare three entry styles on the same history.")
        return

    results_by_name: dict[str, dict[str, Any]] = state["results"]
    best_name = state["best"]
    nran = state.get("n_indicators", "?")
    rmap = state.get("regime_label_map") or {}
    st.success(
        f"**Best fit this run:** {best_name} (highest Sharpe among presets). "
        f"Stack: **{nran}** indicators · cluster map → {rmap or 'n/a'}."
    )

    ohlcv = state["ohlcv"]
    regime_series = state["regime_series"]
    mapped = _mapped_regime_series(ohlcv, regime_series, rmap)
    best_res = results_by_name[best_name]
    last_reg = str(mapped.dropna().iloc[-1]) if len(mapped.dropna()) else "—"
    if last_reg in ("nan", "NaT", "None"):
        last_reg = "—"
    reg_colors = {
        "TREND_UP": ("#22c55e", "Bullish bias"),
        "TREND_DN": ("#ef4444", "Bearish bias"),
        "RANGE": ("#94a3b8", "Range / chop"),
    }
    rc, rlabel = reg_colors.get(last_reg, ("#64748b", "Mixed / unmapped"))

    close_s = ohlcv["close"].astype(float)
    r5 = float(close_s.iloc[-1] / close_s.iloc[-6] - 1) if len(close_s) > 5 else 0.0
    r20 = float(close_s.iloc[-1] / close_s.iloc[-21] - 1) if len(close_s) > 20 else 0.0
    t5, c5 = _thrust_style(r5)
    t20, c20 = _thrust_style(r20)

    st.markdown("##### Market context (at last bar)")
    tc1, tc2, tc3 = st.columns(3)
    tc1.markdown(
        f'<div style="background:{rc}22;border:1px solid {rc}55;border-radius:10px;padding:12px">'
        f'<div style="font-size:0.75rem;opacity:0.85">Regime model</div>'
        f'<div style="font-size:1.35rem;font-weight:700;color:{rc}">{last_reg}</div>'
        f'<div style="font-size:0.8rem;opacity:0.8">{rlabel}</div></div>',
        unsafe_allow_html=True,
    )
    tc2.markdown(
        f'<div style="background:{c5};border-radius:10px;padding:12px">'
        f'<div style="font-size:0.75rem;opacity:0.85">Short thrust (5 bars)</div>'
        f'<div style="font-size:1.35rem;font-weight:700">{t5}</div>'
        f'<div style="font-size:0.8rem;opacity:0.8">Micro momentum</div></div>',
        unsafe_allow_html=True,
    )
    tc3.markdown(
        f'<div style="background:{c20};border-radius:10px;padding:12px">'
        f'<div style="font-size:0.75rem;opacity:0.85">Medium thrust (20 bars)</div>'
        f'<div style="font-size:1.35rem;font-weight:700">{t20}</div>'
        f'<div style="font-size:0.8rem;opacity:0.8">Swing drift</div></div>',
        unsafe_allow_html=True,
    )

    multi = state.get("multi_regime") or {}
    if multi:
        st.markdown("##### Three-window regime stack (k-means, different feature windows)")
        st.caption("SHORT=12 bars · MEDIUM=20 · LONG=40 — same 3-cluster idea, different smoothing.")
        mw1, mw2, mw3 = st.columns(3)
        tier_cols = (mw1, mw2, mw3)
        tier_order = ("SHORT", "MEDIUM", "LONG")
        for col, tag in zip(tier_cols, tier_order):
            block = multi.get(tag) or {}
            rs_t = block.get("series")
            mp = block.get("label_map") or {}
            w = block.get("window", "?")
            if rs_t is None or len(rs_t.dropna()) == 0:
                col.metric(f"{tag} (w={w})", "—")
                continue
            mser = _mapped_regime_series(ohlcv, rs_t, mp)
            lab = str(mser.dropna().iloc[-1]) if len(mser.dropna()) else "—"
            if lab in ("nan", "NaT", "None"):
                lab = "—"
            col.metric(f"{tag} (w={w})", lab)

    st.markdown("##### Visual replay (best preset)")
    st.caption("Shaded bands = mapped regime · triangles = fills/flats · purple line = equity (right axis). Hover a marker for the rule text.")
    fig_replay = _build_visual_replay_figure(
        ohlcv,
        best_res,
        mapped,
        title=f"{state['symbol']} — {best_name.split('(')[0].strip()} preset",
    )
    st.plotly_chart(fig_replay, use_container_width=True)

    n_bars = len(ohlcv)
    min_pb = min(60, n_bars)
    end_ix = st.slider(
        "Playback: show history through this bar (end of window)",
        min_value=min_pb,
        max_value=n_bars,
        value=n_bars,
        key="easy_replay_end_ix",
    )
    if end_ix < n_bars:
        sub_o, sub_m, sub_best = slice_replay_window(ohlcv, mapped, best_res, end_ix)
        fig_pb = _build_visual_replay_figure(
            sub_o,
            sub_best,
            sub_m,
            title=f"Replay through {sub_o.index[-1]} — {best_name.split('(')[0].strip()}",
        )
        st.plotly_chart(fig_pb, use_container_width=True)

    other_presets = [p for p in results_by_name if p != best_name]
    if other_presets:
        compare = st.selectbox("Compare best preset side-by-side with", options=other_presets, key="easy_compare_preset")
        c_left, c_right = st.columns(2)
        with c_left:
            st.caption(f"Best: {best_name.split('(')[0].strip()}")
            st.plotly_chart(
                _build_visual_replay_figure(ohlcv, best_res, mapped, title="Best"),
                use_container_width=True,
            )
        with c_right:
            st.caption(compare.split("(")[0].strip())
            alt = results_by_name[compare]
            st.plotly_chart(
                _build_visual_replay_figure(ohlcv, alt, mapped, title="Alternate"),
                use_container_width=True,
            )

    st.markdown("##### Robustness (return shuffle)")
    st.caption("Monte Carlo on bar returns: if original Sharpe sits far above random permutations, the path is less likely pure luck.")
    if st.button("Run robustness check (best preset)", key="easy_robust"):
        with st.spinner("Bootstrapping…"):
            dist = bootstrap_sharpe_distribution(best_res.get("portfolio_value") or [], n_sims=300)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Original Sharpe", f"{dist['original']:.3f}")
        r2.metric("Shuffle mean", f"{dist['mean']:.3f}")
        r3.metric("5th pct", f"{dist['p05']:.3f}")
        r4.metric("95th pct", f"{dist['p95']:.3f}")
        if dist["original"] < dist["p50"]:
            st.warning("Original Sharpe is **below** the median shuffle — treat edge as fragile.")
        else:
            st.success("Original Sharpe beats the median shuffled path.")

    st.markdown("##### Parameter heatmap (probe stack)")
    st.caption("5 indicators only (RSI/MACD/Bollinger/Dual SMA/Buy&Hold) × same regime tags — ~25 fast backtests.")
    if st.button("Build RSI period × threshold Sharpe grid", key="easy_heat"):
        with st.spinner("Grid search (may take a minute)…"):
            mat = run_rsi_threshold_heatmap(
                ohlcv,
                regime_series,
                rmap,
                deepcopy(DEFAULT_REGIME_BOOSTS),
                symbol=str(state["symbol"]),
            )
        st.session_state["easy_heatmap_df"] = mat
    hm = st.session_state.get("easy_heatmap_df")
    if hm is not None:
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=hm.values,
                x=list(hm.columns),
                y=list(hm.index),
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
            )
        )
        fig_hm.update_layout(
            template="plotly_dark",
            title="Sharpe: rows = signal threshold, cols = RSI period",
            height=420,
            paper_bgcolor="rgba(15,23,42,0.6)",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with st.expander("Trade log & last-bar signal mix", expanded=False):
        te = best_res.get("trade_events") or []
        if te:
            st.dataframe(pd.DataFrame(te), use_container_width=True, hide_index=True)
        else:
            st.caption("No round-trip trades in this window (threshold may be too strict).")
        st.markdown("**Composite** (last bar)")
        st.write(
            f"Blended signal: **{float(best_res.get('composite_last', 0)):.4f}** · "
            f"threshold ±**{float(best_res.get('signal_threshold_used', 0)):.4f}**"
        )
        snap = best_res.get("signal_snapshot_last") or {}
        if snap:
            top = sorted(snap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
            st.dataframe(
                pd.DataFrame([{"indicator": k, "signal": round(v, 4)} for k, v in top]),
                use_container_width=True,
                hide_index=True,
            )

    cols = st.columns(len(PRESETS))
    for i, (label, _) in enumerate(PRESETS):
        r = results_by_name[label]
        cols[i].metric(
            label.split("(")[0].strip(),
            f"{float(r.get('total_return', 0)) * 100:.1f}%",
            help=f"Sharpe {float(r.get('sharpe', 0)):.2f} · max DD {float(r.get('max_drawdown', 0)) * 100:.1f}%",
        )

    fig = go.Figure()
    colors = ("#64748b", "#38bdf8", "#a78bfa")
    for idx, (label, _) in enumerate(PRESETS):
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

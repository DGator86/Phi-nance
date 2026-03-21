"""One-click regime-weighted backtest: full catalog, semantic cluster boosts, API-compat shim."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app_streamlit.config import INDICATOR_SPECS, IndicatorSpec
from phi.backtest.direct import run_direct_backtest
from phi.blending.blender import DEFAULT_REGIME_BOOSTS
from phi.indicators.simple import INDICATOR_COMPUTERS
from phi.regime.train import train_regime_detector

from app_streamlit.easy_mode.constants import LOOKBACK_DAYS, PRIMARY_BACKTEST_SYMBOL
from app_streamlit.easy_mode.data import load_ohlcv

_PRESETS: tuple[tuple[str, float], ...] = (
    ("Cautious (fewer trades)", 0.22),
    ("Balanced", 0.16),
    ("Responsive (more trades)", 0.10),
)


def _default_params_from_spec(spec: IndicatorSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for param, ps in spec.params.items():
        if isinstance(ps, tuple) and len(ps) >= 3:
            out[param] = ps[2]
        elif isinstance(ps, dict) and ps.get("type") == "select":
            opts = list(ps.get("options", []))
            dv = ps.get("default", opts[0] if opts else "")
            out[param] = dv
    return out


def build_auto_indicators_and_weights() -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Same universe as the expert regime workbench: INDICATOR_SPECS ∩ INDICATOR_COMPUTERS."""
    names = sorted(set(INDICATOR_SPECS.keys()) & set(INDICATOR_COMPUTERS.keys()))
    indicators = {
        n: {"enabled": True, "params": _default_params_from_spec(INDICATOR_SPECS[n])} for n in names
    }
    if not indicators:
        raise RuntimeError("No indicators available (catalog intersection empty).")
    n = len(indicators)
    w = round(1.0 / n, 6)
    weights = dict.fromkeys(indicators, w)
    first = next(iter(weights))
    weights[first] = round(w + (1.0 - sum(weights.values())), 6)
    return indicators, weights


def semantic_label_map_for_clusters(ohlcv: pd.DataFrame, regime_series: pd.Series) -> dict[str, str]:
    """Map cluster_* labels to DEFAULT_REGIME_BOOSTS keys using mean return per cluster."""
    rs = regime_series.reindex(ohlcv.index).dropna().astype(str)
    if rs.empty:
        return {}
    close = ohlcv["close"].astype(float)
    rets = close.pct_change()
    labels = sorted(
        rs.unique(),
        key=lambda s: int(s.rsplit("_", 1)[-1]) if str(s).startswith("cluster_") else str(s),
    )
    scored: list[tuple[str, float]] = []
    for lab in labels:
        m = rs == lab
        seg = rets.where(m).dropna()
        scored.append((lab, float(seg.mean()) if len(seg) else 0.0))
    scored.sort(key=lambda x: x[1])
    templates = ("TREND_DN", "RANGE", "TREND_UP")
    n = len(scored)
    if n == 1:
        return {scored[0][0]: "RANGE"}
    if n == 2:
        return {scored[0][0]: "TREND_DN", scored[1][0]: "TREND_UP"}
    out: dict[str, str] = {}
    for i, (lab, _) in enumerate(scored):
        if n == 3:
            out[lab] = templates[i]
        else:
            bucket = min(2, int(3 * i / max(n - 1, 1)))
            out[lab] = templates[bucket]
    return out


def _call_run_direct_backtest(**kwargs: Any) -> tuple[dict[str, Any], Any]:
    """Forward only kwargs accepted by this environment's ``run_direct_backtest``."""
    sig = inspect.signature(run_direct_backtest)
    allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    if kwargs.get("blend_method") == "regime_weighted" and "regime_series" not in allowed:
        filtered["blend_method"] = "weighted_sum"
        for drop in (
            "regime_series",
            "regime_label_map",
            "regime_boosts",
            "regime_detector",
            "regime_detector_params",
        ):
            filtered.pop(drop, None)
    return run_direct_backtest(**filtered)


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
        st.json({"presets": [{"name": n, "signal_threshold": t} for n, t in _PRESETS]})

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

            results_by_name: dict[str, dict[str, Any]] = {}
            for label, thresh in _PRESETS:
                res, _ = _call_run_direct_backtest(
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

    st.markdown("##### Visual replay (best preset)")
    st.caption("Shaded bands = mapped regime · triangles = fills/flats · purple line = equity (right axis). Hover a marker for the rule text.")
    fig_replay = _build_visual_replay_figure(
        ohlcv,
        best_res,
        mapped,
        title=f"{state['symbol']} — {best_name.split('(')[0].strip()} preset",
    )
    st.plotly_chart(fig_replay, use_container_width=True)

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

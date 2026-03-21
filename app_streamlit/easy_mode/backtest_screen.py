"""One-click regime-weighted backtest: full catalog, semantic cluster boosts, API-compat shim."""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

"""
app_streamlit/pages/portfolio_backtest_page.py
================================================

Streamlit page: Multi-Asset Portfolio Backtest

Allows users to:
  1. Select multiple symbols + allocation method
  2. Choose an indicator for signal generation
  3. Run a portfolio-level backtest
  4. View per-asset and aggregate metrics
  5. View the correlation matrix as a heatmap
  6. Download results
"""

from __future__ import annotations

import os

IS_BACKTESTING = os.environ.get("IS_BACKTESTING", "False").lower() == "true"

import numpy as np
import pandas as pd

if not IS_BACKTESTING:
    import streamlit as st

from phinance.backtest.portfolio import (
    PortfolioBacktester,
    PortfolioConfig,
    AllocationMethod,
    run_portfolio_backtest,
)
from phinance.strategies.indicator_catalog import INDICATOR_CATALOG, compute_indicator


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.normal(0, 0.002, n)),
            "high":   close * (1 + abs(rng.normal(0, 0.005, n))),
            "low":    close * (1 - abs(rng.normal(0, 0.005, n))),
            "close":  close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


_SYMBOL_SEEDS = {"SPY": 1, "QQQ": 2, "IWM": 3, "GLD": 4, "TLT": 5, "BTC": 6}


def _get_ohlcv(symbol: str, n: int) -> pd.DataFrame:
    seed = _SYMBOL_SEEDS.get(symbol, hash(symbol) % 100)
    return _make_ohlcv(n=n, seed=seed)


# ── main render function ──────────────────────────────────────────────────────


def render() -> None:
    if IS_BACKTESTING:
        return

    st.header("💼 Multi-Asset Portfolio Backtest")
    st.caption(
        "Run a correlated multi-asset backtest with configurable capital allocation, "
        "position sizing, and signal-driven entry/exit."
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ Portfolio Config")

        symbols = st.multiselect(
            "Symbols",
            options=list(_SYMBOL_SEEDS.keys()),
            default=["SPY", "QQQ"],
        )

        indicator_name = st.selectbox(
            "Signal Indicator",
            options=list(INDICATOR_CATALOG.keys()),
            index=list(INDICATOR_CATALOG.keys()).index("EMA Cross") if "EMA Cross" in INDICATOR_CATALOG else 0,
        )

        allocation = st.selectbox(
            "Allocation Method",
            options=["equal", "risk_parity", "fixed"],
            index=0,
        )

        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=100_000,
            min_value=1_000,
            step=10_000,
        )

        n_bars = st.slider("Data bars", 100, 1000, 252, step=50)

        transaction_cost = st.slider(
            "Transaction cost (round-trip %)",
            0.0, 1.0, 0.1, step=0.05,
        ) / 100.0

        allow_short = st.checkbox("Allow short positions", value=False)

        fixed_weights: dict = {}
        if allocation == "fixed" and symbols:
            st.subheader("Fixed Weights")
            remaining = 1.0
            for sym in symbols[:-1]:
                w = st.slider(f"Weight {sym}", 0.0, 1.0, round(1.0 / len(symbols), 2), step=0.05)
                fixed_weights[sym] = w
                remaining -= w
            fixed_weights[symbols[-1]] = max(0.0, remaining)
            st.caption(f"{symbols[-1]}: {max(0.0, remaining):.2f}")

    # ── Run button ────────────────────────────────────────────────────────────
    if not symbols:
        st.warning("Select at least one symbol.")
        return

    if st.button("🚀 Run Portfolio Backtest", use_container_width=True, type="primary"):
        with st.spinner("Running portfolio backtest…"):
            ohlcv_dict = {sym: _get_ohlcv(sym, n_bars) for sym in symbols}
            signals    = {
                sym: compute_indicator(indicator_name, ohlcv_dict[sym], {}).fillna(0.0)
                for sym in symbols
            }

            result = run_portfolio_backtest(
                ohlcv_dict=ohlcv_dict,
                signals=signals,
                initial_capital=initial_capital,
                allocation=allocation,
                transaction_cost=transaction_cost,
                allow_short=allow_short,
                fixed_weights=fixed_weights if allocation == "fixed" else None,
            )

        # ── Summary cards ─────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return",  f"{result.total_return:.2%}")
        c2.metric("Sharpe Ratio",  f"{result.sharpe:.3f}")
        c3.metric("Max Drawdown",  f"{result.max_drawdown:.2%}")
        c4.metric("Num Trades",    result.num_trades)

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("CAGR",          f"{result.cagr:.2%}")
        c6.metric("Sortino",       f"{result.sortino:.3f}")
        c7.metric("Win Rate",      f"{result.win_rate:.2%}")
        c8.metric("Final Capital", f"${result.final_capital:,.0f}")

        # ── Portfolio equity curve ────────────────────────────────────────
        st.subheader("📈 Portfolio Equity Curve")
        eq_df = pd.DataFrame(
            {"Portfolio NAV": result.portfolio_equity},
            index=range(len(result.portfolio_equity)),
        )
        st.line_chart(eq_df)

        # ── Per-asset results ─────────────────────────────────────────────
        st.subheader("📊 Per-Asset Results")
        asset_rows = []
        for sym, ar in result.asset_results.items():
            asset_rows.append({
                "Symbol":      sym,
                "Allocation":  f"{ar.allocation:.1%}",
                "Return":      f"{ar.total_return:.2%}",
                "Sharpe":      f"{ar.sharpe:.3f}",
                "Max DD":      f"{ar.max_drawdown:.2%}",
                "Win Rate":    f"{ar.win_rate:.2%}",
                "Trades":      ar.num_trades,
            })
        st.dataframe(pd.DataFrame(asset_rows), use_container_width=True)

        # ── Per-asset equity curves ───────────────────────────────────────
        if len(result.asset_results) > 1:
            st.subheader("📈 Per-Asset Equity Curves")
            multi_eq = {}
            for sym, ar in result.asset_results.items():
                if len(ar.equity_curve) > 0:
                    multi_eq[sym] = ar.equity_curve
            if multi_eq:
                min_len = min(len(v) for v in multi_eq.values())
                eq_multi = pd.DataFrame(
                    {sym: v[:min_len] for sym, v in multi_eq.items()}
                )
                st.line_chart(eq_multi)

        # ── Correlation matrix ────────────────────────────────────────────
        if len(symbols) > 1 and result.correlation_matrix.shape[0] > 1:
            st.subheader("🔗 Asset Correlation Matrix")
            corr_df = pd.DataFrame(
                result.correlation_matrix,
                index=symbols[:result.correlation_matrix.shape[0]],
                columns=symbols[:result.correlation_matrix.shape[1]],
            )
            st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                         use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────
        result_dict = result.to_dict()
        result_dict.pop("asset_results", None)
        dl_df = pd.DataFrame([result_dict])
        st.download_button(
            "⬇️ Download portfolio summary CSV",
            data=dl_df.to_csv(index=False),
            file_name="portfolio_result.csv",
            mime="text/csv",
        )

    else:
        st.info("Configure portfolio parameters in the sidebar and click **🚀 Run Portfolio Backtest**.")

    st.divider()
    st.caption("φ Phi-nance · Portfolio Backtest · Multi-Asset Capital Allocation")


if __name__ == "__main__" and not IS_BACKTESTING:
    render()

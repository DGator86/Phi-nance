"""
app_streamlit/pages/evolution_dashboard.py
=============================================

Streamlit page: Strategy Evolution Dashboard

Allows users to:
  1. Configure and run the EvolutionEngine (population, generations, etc.)
  2. Visualise per-generation fitness progress (line chart)
  3. Inspect the best individual (indicators, weights, metrics)
  4. Download the evolution history as CSV
  5. Kick off a Walk-Forward Optimisation on the resulting strategy
"""

from __future__ import annotations

import os

IS_BACKTESTING = os.environ.get("IS_BACKTESTING", "False").lower() == "true"

import numpy as np
import pandas as pd

if not IS_BACKTESTING:
    import streamlit as st

from phinance.agents.evolution_engine import (
    EvolutionEngine,
    EvolutionConfig,
    Individual,
    GenerationResult,
    run_evolution,
)
from phinance.backtest.walk_forward import WalkForwardHarness, WalkForwardConfig


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV for demo purposes."""
    rng = np.random.default_rng(42)
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


def _history_to_df(history: list[GenerationResult]) -> pd.DataFrame:
    rows = [
        {
            "Generation":     gr.generation,
            "Best Fitness":   gr.best_fitness,
            "Mean Fitness":   gr.mean_fitness,
            "Population":     gr.population_size,
            "Deployed":       gr.deployed,
            "Elapsed ms":     gr.elapsed_ms,
        }
        for gr in history
    ]
    return pd.DataFrame(rows)


# ── main render function ──────────────────────────────────────────────────────


def render() -> None:
    if IS_BACKTESTING:
        return

    st.header("🧬 Strategy Evolution Dashboard")
    st.caption(
        "The EvolutionEngine uses a genetic algorithm to continuously evolve "
        "trading strategies without human intervention — proposing, evaluating, "
        "and deploying the fittest strategies each generation."
    )

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ Evolution Config")
        population_size = st.slider("Population size", 4, 30, 8, step=2)
        num_generations  = st.slider("Generations", 1, 20, 4)
        min_indicators   = st.slider("Min indicators / individual", 1, 4, 2)
        max_indicators   = st.slider("Max indicators / individual", 2, 8, 4)
        mutation_rate    = st.slider("Mutation rate", 0.05, 0.8, 0.3, step=0.05)
        deploy_threshold = st.number_input("Fitness threshold for deployment", value=0.5,
                                           min_value=0.0, max_value=50.0, step=0.1)
        dry_run  = st.checkbox("Dry-run (no real deployment)", value=True)
        seed     = st.number_input("Random seed", value=42, min_value=0)
        n_bars   = st.slider("Synthetic data bars", 200, 1000, 300, step=50)

        st.divider()
        st.subheader("🔭 Walk-Forward Config")
        run_wfo  = st.checkbox("Run Walk-Forward after evolution", value=False)
        wfo_is   = st.slider("WFO in-sample bars",  60, 300, 120)
        wfo_oos  = st.slider("WFO out-of-sample bars", 30, 150, 60)
        wfo_step = st.slider("WFO step bars", 20, 120, 60)

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("🚀 Run Evolution", use_container_width=True, type="primary"):
        ohlcv = _make_ohlcv(n_bars)

        cfg = EvolutionConfig(
            population_size=population_size,
            num_generations=num_generations,
            min_indicators=min_indicators,
            max_indicators=max_indicators,
            mutation_rate=mutation_rate,
            deploy_threshold=deploy_threshold,
            dry_run=dry_run,
            random_seed=int(seed),
        )

        with st.spinner(f"Evolving {num_generations} generations × population {population_size}…"):
            engine  = EvolutionEngine(ohlcv=ohlcv, config=cfg)
            history = engine.run()

        best = engine.best_individual
        summary = engine.evolution_summary

        # ── Summary cards ─────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Generations", summary["generations"])
        col2.metric("Best Fitness", f"{summary['best_fitness']:.3f}")
        col3.metric("Final Mean Fitness", f"{summary['final_mean_fitness']:.3f}")
        col4.metric("Auto-Deployments", summary["total_deployments"])

        # ── Fitness chart ─────────────────────────────────────────────────
        hist_df = _history_to_df(history)
        st.subheader("📈 Fitness Progression")
        chart_df = hist_df[["Generation", "Best Fitness", "Mean Fitness"]].set_index("Generation")
        st.line_chart(chart_df)

        # ── Best individual ───────────────────────────────────────────────
        st.subheader("🏆 Best Individual")
        if best:
            b1, b2, b3 = st.columns(3)
            b1.metric("Fitness",    f"{best.fitness:.4f}")
            b2.metric("Sharpe",     f"{best.sharpe:.3f}")
            b3.metric("Max DD",     f"{best.max_drawdown:.2%}")

            b4, b5, b6 = st.columns(3)
            b4.metric("Win Rate",   f"{best.win_rate:.2%}")
            b5.metric("Num Trades", best.num_trades)
            b6.metric("Return",     f"{best.total_return:.2%}")

            st.markdown("**Indicators & Weights**")
            weight_data = [
                {"Indicator": k, "Weight": f"{v:.4f}"}
                for k, v in best.weights.items()
            ]
            if weight_data:
                st.dataframe(pd.DataFrame(weight_data), use_container_width=True)
            else:
                st.info("No indicators in best individual.")
        else:
            st.warning("No best individual found (all fitness = 0).")

        # ── Generation table ──────────────────────────────────────────────
        st.subheader("📋 Generation History")
        st.dataframe(hist_df, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────
        csv = hist_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download history CSV",
            data=csv,
            file_name="evolution_history.csv",
            mime="text/csv",
        )

        # ── Walk-Forward ──────────────────────────────────────────────────
        if run_wfo and best and best.indicators:
            st.subheader("🔭 Walk-Forward Optimisation")
            with st.spinner("Running walk-forward…"):
                wfo_cfg = WalkForwardConfig(
                    is_bars=wfo_is,
                    oos_bars=wfo_oos,
                    step_bars=wfo_step,
                    candidate_names=best.indicators,
                )
                wfo_result = WalkForwardHarness(ohlcv=ohlcv, config=wfo_cfg).run()

            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("WFO Windows",    wfo_result.num_windows)
            wc2.metric("OOS Sharpe",     f"{wfo_result.combined_oos_sharpe:.3f}")
            wc3.metric("Gate Passed",    "✅" if wfo_result.passed_gate else "❌")

            wfo_df = pd.DataFrame([w.to_dict() for w in wfo_result.windows])
            if not wfo_df.empty:
                st.dataframe(wfo_df[["window_id", "best_indicator", "is_sharpe",
                                     "oos_sharpe", "oos_return", "oos_drawdown"]],
                             use_container_width=True)

    else:
        st.info(
            "Configure the evolution parameters in the sidebar and click "
            "**🚀 Run Evolution** to start."
        )

    st.divider()
    st.caption("φ Phi-nance · EvolutionEngine · Genetic Strategy Optimisation")


if __name__ == "__main__" and not IS_BACKTESTING:
    render()

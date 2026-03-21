# Options regime playbook

Phi-nance maps **composite regimes** (trend × volatility, e.g. `BULL_LOW_VOL`) to **approved option structures**, **DTE bands**, **delta bands**, and a **max risk guideline**. The same mapping powers:

- **Trading desk** (Streamlit sidebar page) — full playbook + transition map.
- **Backtest Workbench** — expander next to options fields for quick reference.
- **Options backtest results** — transition map expander; **Attributed PnL by regime** when regime-aware blending is on and a detector produces a bar-aligned series.

## Default data source

Rows are built from `phi.regime.strategy_mapping.REGIME_STRATEGY_MAP`. Transition hints are curated in `phi/options/regime_playbook.py`.

## Custom JSON

Set `PHINANCE_OPTIONS_PLAYBOOK` to a JSON file path. Schema matches `OptionsRegimePlaybook` in `phi/options/regime_playbook.py` (`version`, `regimes`, `transitions` with `from` / `to` keys).

## Price-based quick regime

`quick_detailed_regime_from_ohlcv(ohlcv)` infers a composite label from MA trend + realised vol quantiles — useful for the Trading desk snapshot when no detector is loaded.

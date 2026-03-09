# Regime-Aware Strategy Blending

Regime-aware blending lets you adapt indicator blend weights based on detected market states.

## Concept

Base indicator weights are multiplied by per-regime boosts and then renormalized to sum to 1.0.

Example:

```python
regime_boosts = {
    "Bull": {"RSI": 1.2, "MACD": 0.9},
    "Bear": {"RSI": 0.8, "MACD": 1.1},
}
```

With this setup, RSI is emphasized in Bull regimes while MACD is emphasized in Bear regimes.

## Workflow

1. Train a regime detector in the Streamlit **Regime-Aware Blending** panel.
2. Refresh and select a saved detector from `settings.REGIME_MODELS_DIR`.
3. Choose whether to:
   - run detection on-the-fly during backtest, or
   - reuse pre-computed regime series in the current session.
4. Map detector labels (`state_0`, `state_1`, ...) to friendly names (`Bull`, `Bear`, ...).
5. Enter per-indicator boosts in the matrix editor.
6. Run backtest with `blend_method = regime_weighted`.

## Notes on Interpretation

- Boost values are multipliers, not absolute weights.
- Unspecified indicators default to their base weight multiplier `1.0`.
- Missing regimes on early warmup bars are treated defensively in backtesting.

## Overfitting Risks

- Too many regimes or extreme boosts can overfit historical noise.
- Prefer simple mappings and moderate boosts (e.g., `0.8`–`1.2`) initially.
- Validate on out-of-sample periods and multiple symbols/timeframes.

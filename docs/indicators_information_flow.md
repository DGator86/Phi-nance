# Information Flow Indicators

This document introduces two directional indicators for cross-asset signal discovery:

- **Transfer Entropy** (`transfer_entropy` / `Transfer Entropy`)
- **Granger Causality** (`granger_causality` / `Granger Causality`)

Both are designed for multi-symbol workflows and can be configured in Streamlit under the **Information Flow** category.

## 1) Transfer Entropy

Transfer entropy (TE) estimates how much the past of symbol **X** helps predict the next move of symbol **Y**, beyond Y's own past.

- Input: close prices for `from_symbol` and `to_symbol`.
- Processing:
  1. Convert prices to returns.
  2. Discretize returns into quantile bins (`bins`).
  3. Compute empirical probabilities and apply:
     - `TE(X→Y) = Σ p(y[t+1], y[t], x[t]) * log( p(y[t+1]|y[t],x[t]) / p(y[t+1]|y[t]) )`
- Optional normalization (`normalize=True`) divides by the entropy of target returns in the same window.

### Parameters

- `window`: rolling window size (suggested 30–120)
- `from_symbol`: source symbol (e.g. `SPY`)
- `to_symbol`: target symbol (e.g. `QQQ`)
- `bins`: discretization bins (2–5)
- `normalize`: return entropy-scaled TE

### Interpretation

- Higher TE means stronger directional information flow from `from_symbol` to `to_symbol`.
- Low/near-zero TE means little incremental predictive information.

## 2) Granger Causality

Rolling Granger causality runs a VAR model in each window and tests whether lags of `from_symbol` improve forecasts of `to_symbol`.

### Parameters

- `window`: rolling window size
- `from_symbol`, `to_symbol`
- `maxlags`: VAR lag order
- `threshold`: significance cutoff for binary mode
- `output`: one of
  - `pvalue` (default): lower is stronger evidence
  - `binary`: 1 if `pvalue < threshold`, else 0
  - `confidence`: `1 - pvalue`

### Interpretation

- `pvalue < 0.05` is commonly used as evidence of Granger-causality.
- `binary=1` can be used directly as a rule trigger.
- `confidence` is convenient for blending indicators.

## Streamlit usage

1. Select at least two symbols in the portfolio (`Symbols` input).
2. Add **Transfer Entropy** or **Granger Causality** from *Information Flow*.
3. Set `from_symbol` and `to_symbol` to symbols present in the selected universe.

## Practical notes

- Rolling VAR and TE are more expensive than single-asset indicators.
- For many symbol pairs, start with larger bars (`1H`/`1D`) and moderate windows.
- Warmup periods are `NaN` internally and converted to neutral outputs in wrappers.

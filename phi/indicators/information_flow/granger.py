"""Rolling Granger causality indicators for multi-asset information flow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from phi.indicators.information_flow.transfer_entropy import _extract_symbol_series


def rolling_granger_causality(
    prices: pd.DataFrame,
    from_symbol: str,
    to_symbol: str,
    window: int = 50,
    maxlags: int = 2,
    threshold: float = 0.05,
    output: str = "pvalue",
) -> pd.Series:
    """Rolling Granger causality signal from ``from_symbol`` to ``to_symbol``."""
    if window <= maxlags + 2:
        raise ValueError("window must be larger than maxlags + 2")
    if output not in {"pvalue", "binary", "confidence"}:
        raise ValueError("output must be one of: pvalue, binary, confidence")

    source = _extract_symbol_series(prices, from_symbol).astype(float)
    target = _extract_symbol_series(prices, to_symbol).astype(float)
    returns = pd.concat(
        [target.pct_change().rename(to_symbol), source.pct_change().rename(from_symbol)],
        axis=1,
    )

    out = pd.Series(np.nan, index=prices.index, dtype=float)
    for i in range(window - 1, len(prices)):
        start = i - window + 1
        window_df = returns.iloc[start : i + 1].dropna()
        if len(window_df) <= maxlags + 2:
            continue
        try:
            model = VAR(window_df)
            fitted = model.fit(maxlags=maxlags)
            test = fitted.test_causality(caused=to_symbol, causing=[from_symbol], kind="f")
            pvalue = float(test.pvalue)
        except Exception:
            pvalue = np.nan

        if output == "binary":
            out.iloc[i] = 1.0 if np.isfinite(pvalue) and pvalue < threshold else 0.0
        elif output == "confidence":
            out.iloc[i] = np.nan if not np.isfinite(pvalue) else float(1.0 - np.clip(pvalue, 0.0, 1.0))
        else:
            out.iloc[i] = pvalue

    return out

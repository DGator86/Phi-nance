"""
phinance.strategies.base
========================

Abstract base class for all Phi-nance indicator strategies.

Every indicator must subclass ``BaseIndicator`` and implement:

  ``compute(df, **params) -> pd.Series``
    Normalised signal in the range [-1, 1].
    +1  → strong buy, -1 → strong sell, 0 → neutral.

Optional overrides
------------------
  ``default_params`` — dict of default parameter values
  ``param_grid``     — dict of {param: [grid_values]} for optimisation

Usage
-----
    class MyIndicator(BaseIndicator):
        name = "My Custom"
        default_params = {"period": 20}
        param_grid = {"period": [10, 20, 40]}

        def compute(self, df, period=20, **kwargs):
            ...
            return signal  # pd.Series in [-1, 1]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseIndicator(ABC):
    """Abstract base for all Phi-nance indicator strategies.

    Attributes
    ----------
    name : str
        Human-readable display name used in the catalog and UI.
    default_params : dict
        Default parameter values.
    param_grid : dict
        ``{param_name: [list_of_grid_values]}`` for PhiAI optimisation.
    """

    name: str = "Base"
    default_params: Dict[str, Any] = {}
    param_grid: Dict[str, list] = {}

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params: Any) -> pd.Series:
        """Compute normalised signal from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with columns ``[open, high, low, close, volume]``
            and a DatetimeIndex.
        **params
            Override specific indicator parameters.

        Returns
        -------
        pd.Series
            Signal in [-1, 1] aligned to ``df.index``.
        """
        raise NotImplementedError

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def compute_with_defaults(
        self, df: pd.DataFrame, params: Dict[str, Any] | None = None
    ) -> pd.Series:
        """Merge *params* with ``default_params`` and call ``compute()``.

        Parameters
        ----------
        df : pd.DataFrame
        params : dict, optional
            Caller-supplied parameter overrides.

        Returns
        -------
        pd.Series
        """
        merged = {**self.default_params, **(params or {})}
        try:
            return self.compute(df, **merged)
        except Exception as exc:
            from phinance.exceptions import IndicatorComputationError

            raise IndicatorComputationError(
                f"{self.name}: computation failed — {exc}"
            ) from exc

    @classmethod
    def get_param_grid(cls) -> Dict[str, list]:
        """Return the parameter grid for PhiAI optimisation."""
        return dict(cls.param_grid)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    # ── Signal normalisation helper ───────────────────────────────────────────

    @staticmethod
    def _normalize(s: pd.Series) -> pd.Series:
        """Scale a raw signal series to roughly [-1, 1] using 1%/99% quantiles.

        Maps the range [q01, q99] → [-1, +1] while preserving relative
        ordering.  Suitable for *symmetric* oscillators (MACD histogram,
        spread-based signals) where the 1st and 99th percentiles straddle zero.

        For one-sided indicators use ``_normalize_abs`` instead.
        """
        if s.isna().all():
            return pd.Series(0.0, index=s.index)
        q = s.quantile([0.01, 0.99])
        lo, hi = float(q.iloc[0]), float(q.iloc[1])
        r = hi - lo
        if r == 0:
            return pd.Series(0.0, index=s.index)
        result = ((s - lo) / r - 0.5) * 2
        return result.clip(-1, 1).fillna(0.0)

    @staticmethod
    def _normalize_abs(s: pd.Series) -> pd.Series:
        """Sign-preserving normalisation: divides by the 99th percentile of
        absolute values so that the sign of the raw value is always preserved.

        Use this for trend-following indicators (DualSMA spread, MACD histogram,
        OBV ROC) where positive raw values must map to positive signals.
        """
        if s.isna().all():
            return pd.Series(0.0, index=s.index)
        abs_99 = float(s.abs().quantile(0.99))
        if abs_99 == 0:
            return pd.Series(0.0, index=s.index)
        return (s / abs_99).clip(-1.0, 1.0).fillna(0.0)

"""
Projection Engine — Regime-conditioned AR(1) mixture model.

For each of the 8 collapsed regime bins r:

  x_{t+1}^{(r)} = μ_r + φ_r · (x_t - μ_r) + β_r · Δx_t

Mixture expectation (mean of mixture):

  E[x_{t+1}]   = Σ_r P_t(r) · x_{t+1}^{(r)}

Mixture variance (law of total variance):

  Var[x_{t+1}] = Σ_r P_t(r) · σ_r²                          (within-regime)
               + Σ_r P_t(r) · (x_{t+1}^{(r)} - E[x_{t+1}])² (between-regime)

The projection is applied independently to each indicator signal.
Each indicator's type determines the inverse transform applied after projection:

  Type A — bounded:  inverse logit → tanh space maintained
  Type B — unbounded: raw AR(1) output
  Type C — discrete:  project flip probability with clip [0,1]
  Type D — price:     raw AR(1) output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


REGIME_ORDER = [
    "TREND_UP", "TREND_DN",
    "RANGE",
    "BREAKOUT_UP", "BREAKOUT_DN",
    "EXHAUST_REV",
    "LOWVOL", "HIGHVOL",
]


class ProjectionEngine:
    """
    Regime-conditioned AR(1) projection of indicator signals.

    Parameters
    ----------
    config : dict — the 'projection' sub-dict from config.yaml

    Usage
    -----
    >>> proj = ProjectionEngine(cfg['projection'])
    >>> result = proj.project(signals_df, regime_probs_df, indicator_types)
    # Returns dict with 'expected', 'variance' DataFrames (same shape as signals_df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg    = config
        self._mu    = self._extract_param("mu")
        self._phi   = self._extract_param("phi")
        self._beta  = self._extract_param("beta")
        self._sigma = self._extract_param("sigma")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        signals: pd.DataFrame,
        regime_probs: pd.DataFrame,
        indicator_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Project each signal forward one step under the regime mixture.

        Parameters
        ----------
        signals       : (T, K) DataFrame of normalized indicator signals
        regime_probs  : (T, 8) DataFrame of regime probabilities (linear, sum=1)
        indicator_types : {indicator_name: type_char} — optional type per indicator
                          Defaults to type B if not provided.

        Returns
        -------
        dict with keys:
          'expected'  — (T, K) expected value at t+1 under mixture
          'variance'  — (T, K) mixture variance at t+1
        """
        if indicator_types is None:
            indicator_types = {}

        expected = pd.DataFrame(
            np.nan, index=signals.index, columns=signals.columns
        )
        variance = pd.DataFrame(
            np.nan, index=signals.index, columns=signals.columns
        )

        # Regime probability matrix: (T, 8)
        P = self._align_probs(regime_probs)

        for col in signals.columns:
            x  = signals[col].values.astype(np.float64)   # (T,)
            dx = np.concatenate([[0.0], np.diff(x)])       # Δx_t

            ind_type = indicator_types.get(col, "B")
            exp_col, var_col = self._project_single(x, dx, P, ind_type)

            expected[col] = exp_col
            variance[col] = var_col

        return {"expected": expected, "variance": variance}

    def project_scalar(
        self,
        x_t: float,
        dx_t: float,
        regime_probs: Dict[str, float],
        indicator_type: str = "B",
    ) -> Dict[str, float]:
        """
        Single-bar scalar projection (streaming mode).

        Parameters
        ----------
        x_t          : current signal value
        dx_t         : current signal delta (x_t - x_{t-1})
        regime_probs : {regime_name: probability}
        indicator_type : 'A' | 'B' | 'C' | 'D'

        Returns
        -------
        {'expected': float, 'variance': float}
        """
        P = np.array(
            [regime_probs.get(r, 0.0) for r in REGIME_ORDER], dtype=np.float64
        ).reshape(1, 8)
        x  = np.array([x_t])
        dx = np.array([dx_t])
        exp_arr, var_arr = self._project_single(x, dx, P, indicator_type)
        return {"expected": float(exp_arr[0]), "variance": float(var_arr[0])}

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def _project_single(
        self,
        x:  np.ndarray,   # (T,)
        dx: np.ndarray,   # (T,)
        P:  np.ndarray,   # (T, 8)
        ind_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-bar expected value and variance for one signal."""
        T = len(x)

        # Per-regime projected value: x_{t+1}^{(r)} = μ_r + φ_r*(x_t-μ_r) + β_r*Δx_t
        # Shape: (T, 8)
        mu_r   = self._mu     # (8,)
        phi_r  = self._phi    # (8,)
        beta_r = self._beta   # (8,)
        sig_r  = self._sigma  # (8,)

        x_t_col  = x[:, None]   # (T, 1)
        dx_t_col = dx[:, None]  # (T, 1)

        proj_r = mu_r + phi_r * (x_t_col - mu_r) + beta_r * dx_t_col  # (T, 8)

        # Apply inverse type transform if bounded (Type A / C)
        if ind_type in ("A", "C"):
            proj_r = np.clip(proj_r, -1.0 + 1e-6, 1.0 - 1e-6)

        # Mixture mean E[x_{t+1}] = Σ_r P_r * x_{t+1}^{(r)}
        exp_val = (P * proj_r).sum(axis=1)  # (T,)

        # Mixture variance = within-regime + between-regime
        within  = (P * sig_r**2).sum(axis=1)                    # (T,)
        between = (P * (proj_r - exp_val[:, None])**2).sum(axis=1)  # (T,)
        var_val = within + between

        return exp_val, var_val

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _align_probs(self, regime_probs: pd.DataFrame) -> np.ndarray:
        """Ensure regime_probs has exactly REGIME_ORDER columns, return array."""
        cols = []
        for r in REGIME_ORDER:
            if r in regime_probs.columns:
                cols.append(regime_probs[r].values.astype(np.float64))
            else:
                cols.append(np.zeros(len(regime_probs)))
        P = np.column_stack(cols)   # (T, 8)
        # Normalize rows
        row_sums = P.sum(axis=1, keepdims=True).clip(min=1e-15)
        return P / row_sums

    def _extract_param(self, key: str) -> np.ndarray:
        """Extract AR(1) parameter vector (length 8) in REGIME_ORDER order."""
        regimes = self.cfg["regimes"]
        return np.array(
            [regimes[r][key] for r in REGIME_ORDER], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Parameter access (for AI tuning)
    # ------------------------------------------------------------------

    def get_params_dict(self) -> Dict[str, Any]:
        """Return current AR(1) parameters as a plain dict (AI-tunable)."""
        return {
            r: {
                "mu":    float(self._mu[i]),
                "phi":   float(self._phi[i]),
                "beta":  float(self._beta[i]),
                "sigma": float(self._sigma[i]),
            }
            for i, r in enumerate(REGIME_ORDER)
        }

    def set_params_dict(self, params: Dict[str, Dict[str, float]]) -> None:
        """Update AR(1) parameters from a plain dict (AI tuning interface)."""
        for i, r in enumerate(REGIME_ORDER):
            if r in params:
                self._mu[i]    = params[r].get("mu",    self._mu[i])
                self._phi[i]   = params[r].get("phi",   self._phi[i])
                self._beta[i]  = params[r].get("beta",  self._beta[i])
                self._sigma[i] = params[r].get("sigma", self._sigma[i])

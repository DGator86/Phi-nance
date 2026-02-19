"""
Gamma Surface — Options dealer GEX (Gamma Exposure) profile.

Computes net dealer gamma exposure by strike from an options chain,
then derives regime features from the GEX landscape:

  gamma_wall_distance  — normalized distance to nearest |GEX| peak
                         (positive = wall above spot, negative = below)
  gamma_net            — net total GEX at spot (positive = pinning,
                         negative = amplifying / dealers short gamma)
  gamma_expiry_days    — DTE of the dominant gamma wall
  gex_flip_zone        — 1 if spot is near a GEX zero-crossing (unstable)

GEX sign convention (standard dealer perspective):
  Calls → dealer is short calls → short gamma → +GEX per strike
  Puts  → dealer is long puts  → long gamma  → −GEX per strike
  Net positive GEX → dealers long gamma → pin (absorb moves)
  Net negative GEX → dealers short gamma → amplify (chase moves)

Integration
-----------
  >>> gs = GammaSurface(cfg['gamma'])
  >>> chain_df = fetcher.options_chain('AAPL')
  >>> spot = 175.00
  >>> features = gs.compute_features(chain_df, spot)
  # {'gamma_wall_distance': -0.012, 'gamma_net': 0.43, ...}
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GammaSurface
# ──────────────────────────────────────────────────────────────────────────────

class GammaSurface:
    """
    Computes dealer GEX profile from an options chain and extracts
    four regime features for injection into the taxonomy engine.

    Parameters
    ----------
    config : dict — the 'gamma' sub-dict from config.yaml

    Usage
    -----
    >>> gs = GammaSurface(cfg['gamma'])
    >>> chain_df = fetcher.options_chain('AAPL')
    >>> features = gs.compute_features(chain_df, spot_price)
    """

    #: Names of features produced — must match energy keys in config.yaml
    FEATURE_NAMES = [
        "gamma_wall_distance",
        "gamma_net",
        "gamma_expiry_days",
        "gex_flip_zone",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg              = config
        self.kernel_width_pct = float(config.get("kernel_width_pct",  0.005))
        self.min_oi           = int(  config.get("min_oi",            100))
        self.max_dte          = int(  config.get("max_dte",           60))
        self.flip_threshold   = float(config.get("gex_flip_threshold", 0.10))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(
        self,
        chain_df: pd.DataFrame,
        spot: float,
    ) -> Dict[str, float]:
        """
        Compute all gamma-derived regime features from an options chain.

        Parameters
        ----------
        chain_df : DataFrame from AlphaVantageFetcher.options_chain().
                   Expected columns (case-insensitive): strike, expiration,
                   optiontype, openinterest, gamma.
        spot     : current underlying mid-price.

        Returns
        -------
        dict with keys: gamma_wall_distance, gamma_net,
                        gamma_expiry_days, gex_flip_zone.
        """
        if chain_df is None or chain_df.empty or spot <= 0:
            return self._zero_features()

        try:
            gex_profile = self._compute_gex_profile(chain_df, spot)
            if gex_profile.empty:
                return self._zero_features()
            smoothed = self._smooth_surface(gex_profile, spot)
            return self._extract_features(smoothed, gex_profile, chain_df, spot)
        except Exception as exc:
            logger.warning("GammaSurface.compute_features failed: %s", exc)
            return self._zero_features()

    # ------------------------------------------------------------------
    # GEX computation
    # ------------------------------------------------------------------

    def _compute_gex_profile(
        self,
        chain_df: pd.DataFrame,
        spot: float,
    ) -> pd.Series:
        """
        Compute net dealer GEX at each strike.

        GEX(K) = Σ [ dealer_sign × gamma × OI × 100 × spot ]
          Dealer sign: +1 for calls (dealer short → short gamma)
                       -1 for puts  (dealer long  → long gamma)
        """
        df = chain_df.copy()
        df.columns = [str(c).lower().replace(" ", "_").strip() for c in df.columns]

        # ── Resolve required column names ──────────────────────────────
        strike_col = self._find_col(df, ["strike"])
        gamma_col  = self._find_col(df, ["gamma"])
        oi_col     = self._find_col(df, ["openinterest", "open_interest", "oi"])
        type_col   = self._find_col(df, ["optiontype", "type", "option_type", "cp_flag"])
        exp_col    = self._find_col(df, ["expiration", "expiry", "expiration_date"])

        if strike_col is None or gamma_col is None or oi_col is None:
            logger.debug("GammaSurface: missing required columns in chain_df")
            return pd.Series(dtype=float)

        # ── Filter by minimum OI ───────────────────────────────────────
        df["_oi"] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
        df = df[df["_oi"] >= self.min_oi].copy()

        # ── Filter by DTE ──────────────────────────────────────────────
        if exp_col is not None:
            today = date.today()
            df["_dte"] = df[exp_col].astype(str).map(
                lambda s: max(0, (pd.Timestamp(s).date() - today).days)
                if s not in ("", "nan", "None") else 999
            )
            df = df[df["_dte"] <= self.max_dte].copy()

        if df.empty:
            return pd.Series(dtype=float)

        # ── Dealer sign ────────────────────────────────────────────────
        if type_col is not None:
            df["_sign"] = (
                df[type_col].astype(str).str.lower().str.strip()
                .map({"call": 1.0, "put": -1.0, "c": 1.0, "p": -1.0})
                .fillna(0.0)
            )
        else:
            df["_sign"] = 1.0  # fallback: treat all as calls

        # ── GEX per option ─────────────────────────────────────────────
        df["_gamma"] = pd.to_numeric(df[gamma_col], errors="coerce").fillna(0).clip(lower=0)
        df["_strike"] = pd.to_numeric(df[strike_col], errors="coerce")
        df = df.dropna(subset=["_strike"])

        df["_gex"] = df["_sign"] * df["_gamma"] * df["_oi"] * 100.0 * spot

        return df.groupby("_strike")["_gex"].sum()

    def _smooth_surface(
        self,
        gex_by_strike: pd.Series,
        spot: float,
    ) -> pd.Series:
        """Apply gaussian kernel smoothing across the GEX strike profile."""
        if gex_by_strike.empty:
            return gex_by_strike

        strikes  = gex_by_strike.index.astype(float).values
        gex_vals = gex_by_strike.values.astype(float)
        sigma    = spot * self.kernel_width_pct
        smoothed = np.zeros_like(gex_vals)

        for i, k in enumerate(strikes):
            weights    = np.exp(-0.5 * ((strikes - k) / (sigma + 1e-15)) ** 2)
            smoothed[i] = np.dot(weights, gex_vals) / (weights.sum() + 1e-15)

        return pd.Series(smoothed, index=gex_by_strike.index)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        smoothed: pd.Series,
        raw_gex: pd.Series,
        chain_df: pd.DataFrame,
        spot: float,
    ) -> Dict[str, float]:
        strikes = smoothed.index.astype(float).values
        gex     = smoothed.values.astype(float)
        gex_max = float(np.abs(gex).max()) if len(gex) > 0 else 1.0

        # ── gamma_net: interpolated net GEX at spot ───────────────────
        gamma_net_raw = float(np.interp(spot, strikes, gex))
        gamma_net     = float(np.clip(gamma_net_raw / (gex_max + 1e-15), -1.0, 1.0))

        # ── gamma_wall_distance: nearest dominant |GEX| peak ─────────
        above_mask = strikes >= spot
        below_mask = strikes <  spot

        wall_above = self._find_wall(strikes[above_mask], gex[above_mask])
        wall_below = self._find_wall(strikes[below_mask], gex[below_mask])

        if wall_above is not None and wall_below is not None:
            dist_above = (wall_above - spot) / (spot + 1e-10)
            dist_below = (spot - wall_below)  / (spot + 1e-10)
            gamma_wall_distance = (
                dist_above if dist_above <= dist_below else -dist_below
            )
        elif wall_above is not None:
            gamma_wall_distance = (wall_above - spot) / (spot + 1e-10)
        elif wall_below is not None:
            gamma_wall_distance = -(spot - wall_below) / (spot + 1e-10)
        else:
            gamma_wall_distance = 0.0

        # ── gamma_expiry_days: DTE of the most dominant expiry ────────
        gamma_expiry_days = self._dominant_expiry_days(chain_df, spot)

        # ── gex_flip_zone: near a GEX zero-crossing ──────────────────
        flip_strikes = self._find_zero_crossings(strikes, gex)
        near_flip = any(
            abs(k - spot) / (spot + 1e-10) < self.flip_threshold
            for k in flip_strikes
        )
        gex_flip_zone = 1.0 if near_flip else 0.0

        return {
            "gamma_wall_distance": float(np.clip(gamma_wall_distance, -0.20, 0.20)),
            "gamma_net":           float(gamma_net),
            "gamma_expiry_days":   float(gamma_expiry_days),
            "gex_flip_zone":       float(gex_flip_zone),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_wall(
        strikes: np.ndarray,
        gex: np.ndarray,
    ) -> Optional[float]:
        """Strike with the highest absolute GEX in a subset."""
        if len(strikes) == 0:
            return None
        idx = int(np.abs(gex).argmax())
        return float(strikes[idx])

    @staticmethod
    def _find_zero_crossings(
        strikes: np.ndarray,
        gex: np.ndarray,
    ) -> List[float]:
        """Return linearly-interpolated strikes where GEX changes sign."""
        crossings: List[float] = []
        for i in range(len(gex) - 1):
            if gex[i] * gex[i + 1] < 0:
                t = gex[i] / (gex[i] - gex[i + 1])
                crossings.append(float(strikes[i] + t * (strikes[i + 1] - strikes[i])))
        return crossings

    def _dominant_expiry_days(
        self,
        chain_df: pd.DataFrame,
        spot: float,
    ) -> float:
        """DTE of the expiry with the greatest total |GEX|."""
        df = chain_df.copy()
        df.columns = [str(c).lower().replace(" ", "_").strip() for c in df.columns]

        exp_col   = self._find_col(df, ["expiration", "expiry", "expiration_date"])
        gamma_col = self._find_col(df, ["gamma"])
        oi_col    = self._find_col(df, ["openinterest", "open_interest", "oi"])

        if exp_col is None or gamma_col is None or oi_col is None:
            return 30.0

        today = date.today()
        df["_dte"] = df[exp_col].astype(str).map(
            lambda s: max(0, (pd.Timestamp(s).date() - today).days)
            if s not in ("", "nan", "None") else 999
        )
        df = df[df["_dte"].between(0, self.max_dte)].copy()

        if df.empty:
            return 30.0

        df["_gex_abs"] = (
            pd.to_numeric(df[gamma_col], errors="coerce").fillna(0).clip(lower=0)
            * pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
        )
        agg = df.groupby("_dte")["_gex_abs"].sum()
        if agg.empty:
            return 30.0
        return float(agg.idxmax())

    @staticmethod
    def _find_col(
        df: pd.DataFrame,
        candidates: List[str],
    ) -> Optional[str]:
        """Return the first candidate column name that exists in df."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _zero_features() -> Dict[str, float]:
        return {
            "gamma_wall_distance": 0.0,
            "gamma_net":           0.0,
            "gamma_expiry_days":   30.0,
            "gex_flip_zone":       0.0,
        }

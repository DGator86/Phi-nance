"""
phinance.options.iv_surface
============================

Implied Volatility Surface: builds a 2D (strike × expiry) IV surface from
a collection of option quotes and provides interpolation helpers.

Classes / Functions
-------------------
  IVSurface           — IV surface object (strike × expiry → IV)
  build_iv_surface    — Construct IVSurface from a DataFrame of option quotes
  interpolate_iv      — Linearly interpolate IV for any (strike, T) pair
  smile_for_expiry    — Return IV smile for a single expiry
  term_structure      — Return ATM IV per expiry (term structure)

Input DataFrame columns (build_iv_surface)
------------------------------------------
  strike      : float — strike price
  expiry      : str   — ISO date string  ``"2024-12-20"``
  option_type : str   — ``"call"`` | ``"put"``
  bid         : float
  ask         : float

References
----------
  Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*.
  Wiley.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phinance.options.pricing import implied_volatility


# ── IVPoint dataclass ─────────────────────────────────────────────────────────


@dataclass
class IVPoint:
    """Single calibrated IV point on the surface."""
    strike: float
    expiry: str          # ISO date string
    T: float             # time to expiry in years
    option_type: str
    market_mid: float
    implied_vol: Optional[float]


# ── IVSurface ─────────────────────────────────────────────────────────────────


@dataclass
class IVSurface:
    """
    Implied Volatility surface indexed by expiry and strike.

    Attributes
    ----------
    points : list[IVPoint]   — raw calibrated IV points
    grid   : pd.DataFrame    — pivot table (expiry rows × strike columns)
    spot   : float           — underlying spot price used at build time
    as_of  : str             — snapshot date/time
    """

    points: List[IVPoint] = field(default_factory=list)
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    spot: float = 0.0
    as_of: str = ""

    # ── helpers ───────────────────────────────────────────────────────────────

    def expiries(self) -> List[str]:
        """Sorted list of distinct expiry date strings."""
        return sorted({p.expiry for p in self.points})

    def strikes(self) -> List[float]:
        """Sorted list of distinct strike prices."""
        return sorted({p.strike for p in self.points})

    def smile_for_expiry(self, expiry: str) -> pd.Series:
        """
        Return IV smile for a single expiry as a pd.Series indexed by strike.

        Parameters
        ----------
        expiry : str — ISO date string (must match one in ``self.expiries()``)

        Returns
        -------
        pd.Series — strike → IV (NaN where not calibrated)
        """
        pts = [p for p in self.points if p.expiry == expiry and p.implied_vol is not None]
        if not pts:
            return pd.Series(dtype=float)
        s = pd.Series({p.strike: p.implied_vol for p in pts})
        return s.sort_index()

    def term_structure(self, moneyness: float = 1.0) -> pd.Series:
        """
        ATM (or near-moneyness) IV per expiry — the *volatility term structure*.

        Parameters
        ----------
        moneyness : float — K / S ratio for selecting ATM strike (default 1.0)

        Returns
        -------
        pd.Series — expiry → IV
        """
        target_strike = self.spot * moneyness
        result: Dict[str, float] = {}
        for expiry in self.expiries():
            smile = self.smile_for_expiry(expiry)
            if smile.empty:
                continue
            strikes_arr = np.array(smile.index, dtype=float)
            nearest_idx = int(np.argmin(np.abs(strikes_arr - target_strike)))
            nearest_k = smile.index[nearest_idx]
            result[expiry] = smile[nearest_k]
        return pd.Series(result).sort_index()

    def interpolate(self, strike: float, T: float, method: str = "bilinear") -> Optional[float]:
        """
        Interpolate IV for an arbitrary (strike, T) pair.

        Parameters
        ----------
        strike : float — target strike
        T      : float — time to expiry in years
        method : str   — ``"bilinear"`` (default) | ``"nearest"``

        Returns
        -------
        float — interpolated IV, or None if not possible
        """
        if self.grid.empty:
            return None
        return _interpolate_on_grid(self.grid, strike, T, method)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all IVPoints as a tidy DataFrame."""
        return pd.DataFrame(
            [
                {
                    "strike": p.strike,
                    "expiry": p.expiry,
                    "T": p.T,
                    "option_type": p.option_type,
                    "market_mid": p.market_mid,
                    "implied_vol": p.implied_vol,
                }
                for p in self.points
            ]
        )


# ── Surface builder ───────────────────────────────────────────────────────────


def build_iv_surface(
    quotes: pd.DataFrame,
    spot: float,
    r: float = 0.05,
    as_of: Optional[str] = None,
    reference_date: Optional[date] = None,
) -> IVSurface:
    """
    Build an IVSurface from a DataFrame of option quotes.

    Parameters
    ----------
    quotes : pd.DataFrame
        Must contain columns: ``strike``, ``expiry``, ``option_type``,
        ``bid``, ``ask``.
    spot   : float — underlying spot price
    r      : float — risk-free rate for IV calculation
    as_of  : str, optional — snapshot label
    reference_date : date, optional — used to compute T (defaults to today)

    Returns
    -------
    IVSurface
    """
    required = {"strike", "expiry", "option_type", "bid", "ask"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes DataFrame missing columns: {missing}")

    ref = reference_date or date.today()
    pts: List[IVPoint] = []

    for _, row in quotes.iterrows():
        try:
            expiry_date = date.fromisoformat(str(row["expiry"])[:10])
            T = (expiry_date - ref).days / 365.0
        except Exception:
            continue

        if T <= 0:
            continue

        bid = float(row["bid"] or 0.0)
        ask = float(row["ask"] or 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else max(bid, ask)
        if mid <= 0:
            continue

        strike = float(row["strike"])
        opt_type = str(row["option_type"]).lower()
        if opt_type not in ("call", "put"):
            continue

        iv = implied_volatility(
            option_price=mid,
            S=spot,
            K=strike,
            T=T,
            r=r,
            option_type=opt_type,
        )

        pts.append(
            IVPoint(
                strike=strike,
                expiry=str(row["expiry"])[:10],
                T=T,
                option_type=opt_type,
                market_mid=mid,
                implied_vol=iv,
            )
        )

    # Build pivot grid  (expiry → row, strike → column, value = mean IV)
    grid = _build_grid(pts)

    return IVSurface(
        points=pts,
        grid=grid,
        spot=spot,
        as_of=as_of or str(pd.Timestamp.utcnow()),
    )


# ── Convenience wrappers ──────────────────────────────────────────────────────


def smile_for_expiry(surface: IVSurface, expiry: str) -> pd.Series:
    """Delegate to ``surface.smile_for_expiry(expiry)``."""
    return surface.smile_for_expiry(expiry)


def term_structure(surface: IVSurface, moneyness: float = 1.0) -> pd.Series:
    """Delegate to ``surface.term_structure(moneyness)``."""
    return surface.term_structure(moneyness)


def interpolate_iv(
    surface: IVSurface,
    strike: float,
    T: float,
    method: str = "bilinear",
) -> Optional[float]:
    """Delegate to ``surface.interpolate(strike, T, method)``."""
    return surface.interpolate(strike, T, method)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_grid(pts: List[IVPoint]) -> pd.DataFrame:
    """Pivot IVPoints into a (expiry × strike) grid."""
    if not pts:
        return pd.DataFrame()

    rows = []
    for p in pts:
        if p.implied_vol is not None:
            rows.append({"expiry": p.expiry, "T": p.T, "strike": p.strike, "iv": p.implied_vol})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Use mean when multiple option types share the same (expiry, strike)
    pivot = df.groupby(["T", "strike"])["iv"].mean().unstack("strike")
    pivot.columns = pd.Index([float(c) for c in pivot.columns], name="strike")
    pivot.index.name = "T"
    return pivot.sort_index().sort_index(axis=1)


def _interpolate_on_grid(
    grid: pd.DataFrame,
    strike: float,
    T: float,
    method: str = "bilinear",
) -> Optional[float]:
    """Bilinear / nearest interpolation on the pivot grid."""
    if grid.empty:
        return None

    Ts = np.array(grid.index, dtype=float)
    Ks = np.array(grid.columns, dtype=float)

    if method == "nearest":
        t_idx = int(np.argmin(np.abs(Ts - T)))
        k_idx = int(np.argmin(np.abs(Ks - strike)))
        val = grid.iloc[t_idx, k_idx]
        return float(val) if pd.notna(val) else None

    # Bilinear: find bracketing rows/cols
    t_lo_idx, t_hi_idx = _bracket(Ts, T)
    k_lo_idx, k_hi_idx = _bracket(Ks, strike)

    t_lo, t_hi = Ts[t_lo_idx], Ts[t_hi_idx]
    k_lo, k_hi = Ks[k_lo_idx], Ks[k_hi_idx]

    f_ll = float(grid.iloc[t_lo_idx, k_lo_idx])
    f_lh = float(grid.iloc[t_lo_idx, k_hi_idx])
    f_hl = float(grid.iloc[t_hi_idx, k_lo_idx])
    f_hh = float(grid.iloc[t_hi_idx, k_hi_idx])

    if any(np.isnan([f_ll, f_lh, f_hl, f_hh])):
        # Fall back to nearest
        return _interpolate_on_grid(grid, strike, T, method="nearest")

    wt = (T - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0
    wk = (strike - k_lo) / (k_hi - k_lo) if k_hi != k_lo else 0.0

    val = (
        (1 - wt) * (1 - wk) * f_ll
        + (1 - wt) * wk * f_lh
        + wt * (1 - wk) * f_hl
        + wt * wk * f_hh
    )
    return float(val)


def _bracket(arr: np.ndarray, val: float) -> Tuple[int, int]:
    """Return (lo_idx, hi_idx) that bracket val in a sorted array."""
    n = len(arr)
    if val <= arr[0]:
        return 0, 0
    if val >= arr[-1]:
        return n - 1, n - 1
    idx = int(np.searchsorted(arr, val, side="right")) - 1
    return idx, min(idx + 1, n - 1)

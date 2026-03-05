"""Implied volatility surface construction and interpolation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import QhullError

from phi.data.cache import get_cached_options


@dataclass
class IVPoint:
    strike: float
    expiry: float
    iv: float


class IVSurface:
    """2D IV surface over strike × time-to-expiry (in years)."""

    def __init__(self, chain_data: pd.DataFrame):
        self.min_strike = 0.0
        self.max_strike = 0.0
        self.min_expiry = 0.0
        self.max_expiry = 0.0
        self._linear = None
        self._nearest = None
        self.build_surface(chain_data)

    def build_surface(self, chain_data: pd.DataFrame) -> None:
        df = _normalize_chain_data(chain_data)
        self.min_strike = float(df["strike"].min())
        self.max_strike = float(df["strike"].max())
        self.min_expiry = float(df["expiry"].min())
        self.max_expiry = float(df["expiry"].max())

        points = np.column_stack((df["strike"].to_numpy(dtype=float), df["expiry"].to_numpy(dtype=float)))
        values = df["iv"].to_numpy(dtype=float)
        try:
            self._linear = LinearNDInterpolator(points, values)
        except QhullError:
            self._linear = None
        self._nearest = NearestNDInterpolator(points, values)

    def get_iv(self, strike: float, expiry: float) -> float:
        strike_c = float(np.clip(strike, self.min_strike, self.max_strike))
        expiry_c = float(np.clip(expiry, self.min_expiry, self.max_expiry))
        iv = self._linear(strike_c, expiry_c) if self._linear is not None else np.nan
        if iv is None or (isinstance(iv, float) and np.isnan(iv)) or np.isnan(np.asarray(iv)).any():
            iv = self._nearest(strike_c, expiry_c)
        out = float(np.asarray(iv).item())
        return max(out, 1e-4)

    @classmethod
    def from_cached_options(cls, symbol: str, expiries: list[str], today: Optional[pd.Timestamp] = None) -> "IVSurface":
        rows = []
        today_ts = pd.Timestamp(today) if today is not None else pd.Timestamp.utcnow().tz_localize(None)
        for expiry_str in expiries:
            chain = get_cached_options(symbol, expiry_str)
            if chain is None or chain.empty:
                continue
            expiry_dt = pd.Timestamp(expiry_str)
            t = max((expiry_dt - today_ts).days / 365.0, 1 / 365)
            grouped = chain.groupby("strike", as_index=False)
            for _, part in grouped:
                iv_cols = [c for c in ["impliedVolatility", "iv", "implied_volatility"] if c in part.columns]
                if not iv_cols:
                    continue
                iv = float(part[iv_cols[0]].dropna().mean()) if not part[iv_cols[0]].dropna().empty else np.nan
                if np.isnan(iv):
                    continue
                rows.append({"strike": float(part["strike"].iloc[0]), "expiry": t, "iv": iv})
        if not rows:
            raise ValueError("No cached options data available to build IV surface")
        return cls(pd.DataFrame(rows))


def _normalize_chain_data(chain_data: pd.DataFrame) -> pd.DataFrame:
    required = {"strike", "expiry"}
    if not required.issubset(chain_data.columns):
        missing = sorted(required - set(chain_data.columns))
        raise ValueError(f"chain_data missing required columns: {missing}")

    df = chain_data.copy()
    if "iv" not in df.columns:
        call_iv = df["call_iv"] if "call_iv" in df.columns else np.nan
        put_iv = df["put_iv"] if "put_iv" in df.columns else np.nan
        if isinstance(call_iv, pd.Series) or isinstance(put_iv, pd.Series):
            df["iv"] = pd.concat([pd.Series(call_iv), pd.Series(put_iv)], axis=1).mean(axis=1)
        else:
            raise ValueError("chain_data must include either 'iv' or call/put IV columns")

    out = df[["strike", "expiry", "iv"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    if out.empty:
        raise ValueError("No usable IV points after cleaning")
    return out

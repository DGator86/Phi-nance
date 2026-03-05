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


class HistoricalIVSurface:
    """Time-aware IV surface with cached snapshots and HV fallback."""

    def __init__(self, symbol: str, start_date, end_date, price_history: Optional[pd.DataFrame] = None, fallback_window: int = 20):
        self.symbol = symbol.upper()
        self.start_date = pd.Timestamp(start_date).normalize()
        self.end_date = pd.Timestamp(end_date).normalize()
        self.snapshots: dict[pd.Timestamp, IVSurface] = {}
        self.fallback_window = int(max(fallback_window, 2))

        self._load_cached_surfaces()
        self._hv_series = self._build_hv_series(price_history)

    def _load_cached_surfaces(self) -> None:
        root = pd.Timestamp(self.start_date)
        from phi.data.cache import _DATA_CACHE_ROOT  # local import to avoid top-level cache dependency cycles
        opt_root = _DATA_CACHE_ROOT / "options" / self.symbol
        if not opt_root.exists():
            return
        for d in opt_root.iterdir():
            if not d.is_dir():
                continue
            try:
                expiry = pd.Timestamp(d.name)
            except Exception:
                continue
            if expiry < root:
                continue
            chain = get_cached_options(self.symbol, d.name)
            if chain is None or chain.empty:
                continue
            iv_col = next((c for c in ["impliedVolatility", "iv", "implied_volatility"] if c in chain.columns), None)
            if iv_col is None or "strike" not in chain.columns:
                continue
            grouped = chain.groupby("strike", as_index=False)[iv_col].mean().rename(columns={iv_col: "iv"})
            grouped["expiry"] = max((expiry - self.start_date).days / 365.0, 1 / 365)
            try:
                self.snapshots[expiry.normalize()] = IVSurface(grouped[["strike", "expiry", "iv"]])
            except ValueError:
                continue

    def _build_hv_series(self, price_history: Optional[pd.DataFrame]) -> pd.Series:
        if price_history is None or price_history.empty or "close" not in price_history.columns:
            return pd.Series(dtype=float)
        close = price_history["close"].astype(float)
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        hv = ret.rolling(self.fallback_window).std().fillna(method="bfill").fillna(0.2) * np.sqrt(252)
        hv.index = pd.to_datetime(hv.index).normalize()
        return hv.clip(lower=0.05, upper=1.5)

    def get_iv(self, dt, strike: float, expiry: float) -> float:
        date_key = pd.Timestamp(dt).normalize()
        if self.snapshots:
            nearest = min(self.snapshots.keys(), key=lambda d: abs((d - date_key).days))
            return self.snapshots[nearest].get_iv(strike, expiry)
        if date_key in self._hv_series.index:
            return float(self._hv_series.loc[date_key])
        if not self._hv_series.empty:
            nearest = min(self._hv_series.index, key=lambda d: abs((d - date_key).days))
            return float(self._hv_series.loc[nearest])
        return 0.2

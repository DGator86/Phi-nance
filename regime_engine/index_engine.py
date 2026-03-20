"""
Index Engine — Calendar-based and macro index impact features.

Computes six scalar features derived from the trading calendar and known
systematic rebalancing patterns.  These are injected into the feature
DataFrame alongside GEX and L2 signals so the taxonomy can detect regimes
driven by index mechanics rather than pure price action.

Features produced
-----------------
  month_end_flag        — 1.0 within the last ``month_end_days`` trading
                          days of a calendar month; 0.0 otherwise.
                          Month-end rebalancing by mutual funds creates
                          predictable buy-side flow.

  quarter_end_flag      — 1.0 within the last ``quarter_end_days`` trading
                          days of a calendar quarter; 0.0 otherwise.
                          Stronger than month_end; pension/ETF rebalancing
                          dominates order flow.

  opex_proximity        — Normalised proximity to the nearest monthly
                          options expiration (3rd Friday of each month).
                          Range [0, 1]: 1.0 = expiration day, decays to 0
                          at ``opex_horizon_days`` or more away.

  quad_witching_flag    — 1.0 within ``quad_witching_days`` calendar days
                          of a quarterly options expiration (3rd Friday of
                          Mar / Jun / Sep / Dec); 0.0 otherwise.
                          These sessions have the largest index rebalancing
                          and gamma expiration flows.

  fomc_proximity        — Normalised proximity to the nearest estimated
                          FOMC meeting.  Range [0, 1]: 1.0 = meeting day,
                          decays to 0 at ``fomc_horizon_days`` or more.
                          Known 2024–2026 dates are hard-coded; future
                          dates use a ~6.5-week recurrence approximation.

  rebalance_pressure    — Composite score combining the above calendar
                          signals into a single [0, 1] value.  Weights:
                            0.35 × quarter_end_flag
                            0.25 × quad_witching_flag
                            0.20 × opex_proximity
                            0.15 × month_end_flag
                            0.05 × fomc_proximity

Integration
-----------
  >>> from regime_engine.index_engine import IndexEngine
  >>> ie = IndexEngine(cfg.get('index_engine', {}))
  >>> index_features = ie.compute_features(ohlcv.index)
  >>> # {'month_end_flag': 0.0, 'quarter_end_flag': 1.0, ...}
  # Then inject via scanner.py alongside gamma/l2 features.

Notes
-----
All features are computed from the DatetimeIndex of the OHLCV series.
When the index is not timezone-aware or is not a DatetimeIndex, the engine
returns ZERO_FEATURES gracefully.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Known FOMC meeting dates (first day of 2-day meeting; use as the "event day")
# ──────────────────────────────────────────────────────────────────────────────
# Sources: Federal Reserve public calendar.
# Dates beyond the known schedule are approximated by the engine.
_FOMC_KNOWN_DATES: List[date] = [
    # 2024
    date(2024,  1, 30), date(2024,  3, 19), date(2024,  4, 30),
    date(2024,  6, 11), date(2024,  7, 30), date(2024,  9, 17),
    date(2024, 11,  6), date(2024, 12, 17),
    # 2025
    date(2025,  1, 28), date(2025,  3, 18), date(2025,  5,  6),
    date(2025,  6, 17), date(2025,  7, 29), date(2025,  9, 16),
    date(2025, 10, 28), date(2025, 12,  9),
    # 2026
    date(2026,  1, 27), date(2026,  3, 17), date(2026,  4, 28),
    date(2026,  6, 16), date(2026,  7, 28), date(2026,  9, 15),
    date(2026, 10, 27), date(2026, 12,  8),
]

# Average inter-meeting period in calendar days (used when extrapolating)
_FOMC_AVG_SPACING_DAYS: int = 45


# ──────────────────────────────────────────────────────────────────────────────
# Calendar helpers
# ──────────────────────────────────────────────────────────────────────────────

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the date of the n-th occurrence of weekday (0=Mon…6=Sun) in month."""
    first = date(year, month, 1)
    first_weekday = first.weekday()
    day = first.day + (weekday - first_weekday) % 7 + (n - 1) * 7
    return date(year, month, day)


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of a given month (options expiration)."""
    return _nth_weekday_of_month(year, month, 4, 3)  # 4 = Friday


def _is_quad_witching_month(month: int) -> bool:
    """Quarterly options expiration occurs in March, June, September, December."""
    return month in (3, 6, 9, 12)


def _nearest_fomc(d: date) -> date:
    """Return the FOMC date nearest to *d* from the known schedule or approximation."""
    all_dates = list(_FOMC_KNOWN_DATES)

    # Extrapolate future dates beyond known schedule using average spacing
    if all_dates:
        last_known = max(all_dates)
        step = timedelta(days=_FOMC_AVG_SPACING_DAYS)
        candidate = last_known + step
        while candidate <= d + timedelta(days=180):
            all_dates.append(candidate)
            candidate += step

    if not all_dates:
        return d

    return min(all_dates, key=lambda fd: abs((fd - d).days))


def _trading_days_remaining_in_month(d: date) -> int:
    """Rough approximation of trading days left in the current month.

    Uses calendar days × 5/7 heuristic — sufficient for the flag features.
    """
    import calendar
    last_day = calendar.monthrange(d.year, d.month)[1]
    remaining_cal = last_day - d.day
    return max(0, int(round(remaining_cal * 5 / 7)))


# ──────────────────────────────────────────────────────────────────────────────
# IndexEngine
# ──────────────────────────────────────────────────────────────────────────────

class IndexEngine:
    """
    Computes calendar-derived index/macro impact features.

    Parameters
    ----------
    config : dict
        The ``index_engine`` sub-dict from config.yaml.

    Usage
    -----
    >>> ie = IndexEngine(cfg.get('index_engine', {}))
    >>> features = ie.compute_features(ohlcv.index)
    # Returns a dict — broadcast onto feature DataFrame in scanner.py.

    Or for a full time-series (one row per bar):
    >>> feature_df = ie.compute_feature_series(ohlcv.index)
    """

    FEATURE_NAMES: List[str] = [
        "month_end_flag",
        "quarter_end_flag",
        "opex_proximity",
        "quad_witching_flag",
        "fomc_proximity",
        "rebalance_pressure",
    ]

    ZERO_FEATURES: Dict[str, float] = {
        "month_end_flag":     0.0,
        "quarter_end_flag":   0.0,
        "opex_proximity":     0.0,
        "quad_witching_flag": 0.0,
        "fomc_proximity":     0.0,
        "rebalance_pressure": 0.0,
    }

    # Composite weights must sum to 1.0
    _COMPOSITE_WEIGHTS = {
        "quarter_end_flag":   0.35,
        "quad_witching_flag": 0.25,
        "opex_proximity":     0.20,
        "month_end_flag":     0.15,
        "fomc_proximity":     0.05,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.month_end_days      = int(  cfg.get("month_end_days",       3))
        self.quarter_end_days    = int(  cfg.get("quarter_end_days",     5))
        self.opex_horizon_days   = int(  cfg.get("opex_horizon_days",   10))
        self.quad_witching_days  = int(  cfg.get("quad_witching_days",   2))
        self.fomc_horizon_days   = int(  cfg.get("fomc_horizon_days",    5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_features(
        self,
        index: pd.Index,
    ) -> Dict[str, float]:
        """
        Compute features for the **last** bar of the index.

        Returns a single dict of scalar floats for injection as broadcast
        constants, mirroring the pattern used by GammaSurface and
        PolygonL2Client.

        Parameters
        ----------
        index : pd.Index
            DatetimeIndex of the OHLCV series.  Non-DatetimeIndex → ZERO_FEATURES.

        Returns
        -------
        dict with keys matching FEATURE_NAMES.
        """
        try:
            if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
                return dict(self.ZERO_FEATURES)
            last_ts = index[-1]
            d = pd.Timestamp(last_ts).date()
            return self._features_for_date(d)
        except Exception as exc:
            logger.warning("IndexEngine.compute_features failed: %s", exc)
            return dict(self.ZERO_FEATURES)

    def compute_feature_series(
        self,
        index: pd.Index,
    ) -> pd.DataFrame:
        """
        Compute features for **every bar** in the index.

        Unlike the broadcast pattern used by GammaSurface (which injects a
        constant for the whole series), this returns a per-bar DataFrame so
        that the taxonomy can track how calendar pressure evolves intraday.

        Parameters
        ----------
        index : pd.Index — DatetimeIndex of the OHLCV series.

        Returns
        -------
        pd.DataFrame, shape (len(index), len(FEATURE_NAMES)).
        """
        if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
            return pd.DataFrame(
                0.0, index=index, columns=self.FEATURE_NAMES
            )
        try:
            rows = [self._features_for_date(pd.Timestamp(ts).date()) for ts in index]
            return pd.DataFrame(rows, index=index)
        except Exception as exc:
            logger.warning("IndexEngine.compute_feature_series failed: %s", exc)
            return pd.DataFrame(0.0, index=index, columns=self.FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Feature computation for a single date
    # ------------------------------------------------------------------

    def _features_for_date(self, d: date) -> Dict[str, float]:
        """Compute all features for a given calendar date."""

        # ── month_end_flag ────────────────────────────────────────────
        td_left = _trading_days_remaining_in_month(d)
        month_end_flag = 1.0 if td_left <= self.month_end_days else 0.0

        # ── quarter_end_flag ──────────────────────────────────────────
        # Quarter ends: March 31, June 30, September 30, December 31
        # Check if within ``quarter_end_days`` trading days of the quarter end.
        quarter_end_flag = 0.0
        if d.month in (3, 6, 9, 12):
            import calendar as _cal
            last_day = _cal.monthrange(d.year, d.month)[1]
            cal_days_left = last_day - d.day
            td_left_q = max(0, int(round(cal_days_left * 5 / 7)))
            if td_left_q <= self.quarter_end_days:
                quarter_end_flag = 1.0

        # ── opex_proximity ────────────────────────────────────────────
        # 3rd Friday of current month — and look ahead to next month too.
        opex_this = _third_friday(d.year, d.month)
        if d.month == 12:
            opex_next = _third_friday(d.year + 1, 1)
        else:
            opex_next = _third_friday(d.year, d.month + 1)

        days_to_this = abs((opex_this - d).days)
        days_to_next = abs((opex_next - d).days)
        days_to_opex = min(days_to_this, days_to_next)
        opex_proximity = max(0.0, 1.0 - days_to_opex / (self.opex_horizon_days + 1e-10))

        # ── quad_witching_flag ────────────────────────────────────────
        # Quarterly options expiration (3rd Friday of Mar/Jun/Sep/Dec).
        quad_witching_flag = 0.0
        if _is_quad_witching_month(d.month):
            qw_date = _third_friday(d.year, d.month)
            if abs((qw_date - d).days) <= self.quad_witching_days:
                quad_witching_flag = 1.0
        # Also check previous quarter's month if we're in the early days of
        # the following month (e.g. April 1–2 can still carry quad witching flow)
        prev_month = d.month - 1 if d.month > 1 else 12
        prev_year  = d.year if d.month > 1 else d.year - 1
        if _is_quad_witching_month(prev_month):
            prev_qw = _third_friday(prev_year, prev_month)
            if abs((prev_qw - d).days) <= self.quad_witching_days:
                quad_witching_flag = 1.0

        # ── fomc_proximity ────────────────────────────────────────────
        nearest_fomc_date = _nearest_fomc(d)
        days_to_fomc = abs((nearest_fomc_date - d).days)
        fomc_proximity = max(0.0, 1.0 - days_to_fomc / (self.fomc_horizon_days + 1e-10))

        # ── rebalance_pressure ────────────────────────────────────────
        raw = {
            "month_end_flag":     month_end_flag,
            "quarter_end_flag":   quarter_end_flag,
            "opex_proximity":     opex_proximity,
            "quad_witching_flag": quad_witching_flag,
            "fomc_proximity":     fomc_proximity,
        }
        rebalance_pressure = float(sum(
            w * raw[k] for k, w in self._COMPOSITE_WEIGHTS.items()
        ))

        return {
            "month_end_flag":     float(month_end_flag),
            "quarter_end_flag":   float(quarter_end_flag),
            "opex_proximity":     float(opex_proximity),
            "quad_witching_flag": float(quad_witching_flag),
            "fomc_proximity":     float(fomc_proximity),
            "rebalance_pressure": float(rebalance_pressure),
        }

"""
tests/unit/test_options.py
===========================

Unit tests for phinance.options:
  - Black-Scholes pricing (call, put, put-call parity)
  - Implied volatility
  - Option Greeks
  - IV Surface (build, smile, term_structure, interpolate)
  - run_options_backtest
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from tests.fixtures.ohlcv import make_ohlcv

import pandas as pd
from datetime import date, timedelta

from phinance.options.pricing import (
    black_scholes_call,
    black_scholes_put,
    implied_volatility,
    _d1,
    _d2,
    _norm_cdf,
    _norm_pdf,
)
from phinance.options.greeks import OptionsGreeks, compute_greeks
from phinance.options.backtest import run_options_backtest
from phinance.options.iv_surface import (
    IVSurface,
    IVPoint,
    build_iv_surface,
    interpolate_iv,
    smile_for_expiry,
    term_structure,
    _bracket,
    _build_grid,
)


# ─────────────────────────────────────────────────────────────────────────────
#  _norm_cdf / _norm_pdf
# ─────────────────────────────────────────────────────────────────────────────

class TestMathHelpers:

    def test_norm_cdf_at_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-9

    def test_norm_cdf_large_positive(self):
        assert _norm_cdf(10.0) > 0.9999

    def test_norm_cdf_large_negative(self):
        assert _norm_cdf(-10.0) < 0.0001

    def test_norm_cdf_symmetric(self):
        x = 1.2
        assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 1e-12

    def test_norm_pdf_at_zero(self):
        expected = 1 / math.sqrt(2 * math.pi)
        assert abs(_norm_pdf(0.0) - expected) < 1e-12

    def test_norm_pdf_positive(self):
        assert _norm_pdf(0.0) > _norm_pdf(1.0) > _norm_pdf(2.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Black-Scholes call / put
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackScholesPricing:

    # Standard test case: S=100, K=100, T=1, r=0.05, sigma=0.20
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_call_positive(self):
        c = black_scholes_call(self.S, self.K, self.T, self.r, self.sigma)
        assert c > 0

    def test_put_positive(self):
        p = black_scholes_put(self.S, self.K, self.T, self.r, self.sigma)
        assert p > 0

    def test_put_call_parity(self):
        """C - P = S - K * exp(-r*T)  (put-call parity)."""
        c = black_scholes_call(self.S, self.K, self.T, self.r, self.sigma)
        p = black_scholes_put( self.S, self.K, self.T, self.r, self.sigma)
        lhs = c - p
        rhs = self.S - self.K * math.exp(-self.r * self.T)
        assert abs(lhs - rhs) < 1e-8

    def test_call_intrinsic_at_expiry(self):
        """At T=0, call price = max(S-K, 0)."""
        c = black_scholes_call(110.0, 100.0, 0.0, 0.05, 0.20)
        assert abs(c - 10.0) < 1e-9

    def test_put_intrinsic_at_expiry(self):
        p = black_scholes_put(90.0, 100.0, 0.0, 0.05, 0.20)
        assert abs(p - 10.0) < 1e-9

    def test_call_increases_with_underlying(self):
        c1 = black_scholes_call(100, 100, 1, 0.05, 0.20)
        c2 = black_scholes_call(110, 100, 1, 0.05, 0.20)
        assert c2 > c1

    def test_put_decreases_with_underlying(self):
        p1 = black_scholes_put(100, 100, 1, 0.05, 0.20)
        p2 = black_scholes_put(110, 100, 1, 0.05, 0.20)
        assert p2 < p1

    def test_call_increases_with_volatility(self):
        c1 = black_scholes_call(100, 100, 1, 0.05, 0.10)
        c2 = black_scholes_call(100, 100, 1, 0.05, 0.30)
        assert c2 > c1

    def test_atm_call_well_known_benchmark(self):
        """ATM call with S=K=100, T=1, r=0.05, sigma=0.2 ≈ 10.45."""
        c = black_scholes_call(100, 100, 1, 0.05, 0.20)
        assert 9.0 < c < 12.0

    def test_deep_itm_call_approaches_intrinsic(self):
        c = black_scholes_call(200, 100, 1, 0.05, 0.01)
        intrinsic = 200 - 100 * math.exp(-0.05)
        assert abs(c - intrinsic) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
#  Implied Volatility
# ─────────────────────────────────────────────────────────────────────────────

class TestImpliedVolatility:

    def test_roundtrip_call(self):
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25
        price = black_scholes_call(S, K, T, r, sigma)
        iv = implied_volatility(price, S, K, T, r, "call")
        assert iv is not None
        assert abs(iv - sigma) < 1e-4

    def test_roundtrip_put(self):
        S, K, T, r, sigma = 100, 110, 1, 0.05, 0.30
        price = black_scholes_put(S, K, T, r, sigma)
        iv = implied_volatility(price, S, K, T, r, "put")
        assert iv is not None
        assert abs(iv - sigma) < 1e-4

    def test_zero_price_returns_none(self):
        iv = implied_volatility(0.0, 100, 100, 1, 0.05)
        assert iv is None

    def test_expired_returns_none(self):
        iv = implied_volatility(5.0, 100, 100, 0.0, 0.05)
        assert iv is None


# ─────────────────────────────────────────────────────────────────────────────
#  Option Greeks
# ─────────────────────────────────────────────────────────────────────────────

class TestOptionsGreeks:

    def test_call_delta_between_zero_and_one(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "call")
        assert 0.0 < g.delta < 1.0

    def test_put_delta_between_minus_one_and_zero(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "put")
        assert -1.0 < g.delta < 0.0

    def test_gamma_positive(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "call")
        assert g.gamma > 0

    def test_gamma_same_for_call_and_put(self):
        gc = compute_greeks(100, 100, 1, 0.05, 0.20, "call")
        gp = compute_greeks(100, 100, 1, 0.05, 0.20, "put")
        assert abs(gc.gamma - gp.gamma) < 1e-10

    def test_theta_negative_for_long_call(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "call")
        assert g.theta < 0

    def test_vega_positive(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "call")
        assert g.vega > 0

    def test_call_rho_positive(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "call")
        assert g.rho > 0

    def test_put_rho_negative(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20, "put")
        assert g.rho < 0

    def test_to_dict(self):
        g = compute_greeks(100, 100, 1, 0.05, 0.20)
        d = g.to_dict()
        assert set(d.keys()) == {"delta", "gamma", "theta", "vega", "rho"}
        assert all(np.isfinite(v) for v in d.values())

    def test_expired_option_finite_greeks(self):
        g = compute_greeks(100, 100, 0, 0.05, 0.20)
        assert np.isfinite(g.delta)
        assert np.isfinite(g.gamma)

    def test_atm_call_delta_near_half(self):
        """ATM call delta ≈ 0.5 for zero rates and short time."""
        g = compute_greeks(100, 100, 0.01, 0.0, 0.20, "call")
        assert 0.45 < g.delta < 0.55

    def test_deep_itm_call_delta_near_one(self):
        g = compute_greeks(200, 100, 1, 0.05, 0.20, "call")
        assert g.delta > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        g = compute_greeks(50, 100, 1, 0.05, 0.20, "call")
        assert g.delta < 0.05


# ─────────────────────────────────────────────────────────────────────────────
#  run_options_backtest
# ─────────────────────────────────────────────────────────────────────────────

class TestRunOptionsBacktest:

    def test_returns_dict_with_required_keys(self):
        df = make_ohlcv(100)
        result = run_options_backtest(df, symbol="TEST", strategy_type="long_call")
        for key in ("portfolio_value", "total_return", "cagr", "max_drawdown", "sharpe"):
            assert key in result

    def test_portfolio_value_has_correct_length(self):
        df = make_ohlcv(100)
        result = run_options_backtest(df)
        # Length is n+1 (initial + one per bar, or n if start bar not included)
        assert len(result["portfolio_value"]) >= len(df)

    def test_total_return_is_finite(self):
        df = make_ohlcv(100)
        result = run_options_backtest(df)
        assert np.isfinite(result["total_return"])

    def test_long_call_vs_long_put(self):
        """Long call and long put should give different returns on same data."""
        df = make_ohlcv(100)
        r_call = run_options_backtest(df, strategy_type="long_call")
        r_put  = run_options_backtest(df, strategy_type="long_put")
        # They should differ (P&L signs will generally be opposite)
        assert r_call["total_return"] != r_put["total_return"]

    def test_short_df_returns_initial(self):
        df = make_ohlcv(2)
        result = run_options_backtest(df, initial_capital=50_000)
        # Minimal data — just returns initial
        assert len(result["portfolio_value"]) >= 1

    def test_max_drawdown_non_negative(self):
        df = make_ohlcv(100)
        result = run_options_backtest(df)
        assert result["max_drawdown"] <= 0  # drawdown is stored as negative fraction

    def test_sharpe_finite(self):
        df = make_ohlcv(200)
        result = run_options_backtest(df)
        assert np.isfinite(result["sharpe"])


# ─────────────────────────────────────────────────────────────────────────────
#  IV Surface
# ─────────────────────────────────────────────────────────────────────────────

def _make_quotes(spot: float = 100.0, n_strikes: int = 5, n_expiries: int = 3) -> pd.DataFrame:
    """Create synthetic option quote DataFrame for testing."""
    ref = date(2026, 3, 3)
    rows = []
    for days in [30, 60, 90][:n_expiries]:
        expiry = (ref + timedelta(days=days)).isoformat()
        T = days / 365.0
        for dk in range(-n_strikes // 2, n_strikes // 2 + 1):
            strike = round(spot * (1 + dk * 0.05), 1)
            for opt_type in ("call", "put"):
                sigma = 0.20 + abs(dk) * 0.01  # slight vol smile
                price = (
                    black_scholes_call(spot, strike, T, 0.05, sigma)
                    if opt_type == "call"
                    else black_scholes_put(spot, strike, T, 0.05, sigma)
                )
                if price > 0.01:
                    rows.append({
                        "strike": strike,
                        "expiry": expiry,
                        "option_type": opt_type,
                        "bid": price * 0.99,
                        "ask": price * 1.01,
                    })
    return pd.DataFrame(rows)


class TestIVSurface:

    def _surface(self, spot: float = 100.0) -> IVSurface:
        quotes = _make_quotes(spot)
        return build_iv_surface(quotes, spot=spot, r=0.05,
                                reference_date=date(2026, 3, 3))

    # ── build ─────────────────────────────────────────────────────────────────

    def test_build_returns_iv_surface(self):
        surf = self._surface()
        assert isinstance(surf, IVSurface)

    def test_points_not_empty(self):
        surf = self._surface()
        assert len(surf.points) > 0

    def test_all_points_have_finite_iv(self):
        surf = self._surface()
        for p in surf.points:
            if p.implied_vol is not None:
                assert 0.001 < p.implied_vol < 5.0, f"IV out of range: {p.implied_vol}"

    def test_grid_not_empty(self):
        surf = self._surface()
        assert not surf.grid.empty

    def test_spot_stored(self):
        surf = self._surface(spot=150.0)
        assert surf.spot == 150.0

    def test_as_of_set(self):
        surf = self._surface()
        assert len(surf.as_of) > 0

    def test_missing_columns_raises(self):
        with pytest.raises(ValueError, match="missing columns"):
            build_iv_surface(pd.DataFrame({"strike": [100]}), spot=100.0)

    def test_expired_quotes_excluded(self):
        """Quotes with expiry in the past should produce no IVPoints."""
        past_expiry = (date(2026, 3, 3) - timedelta(days=10)).isoformat()
        quotes = pd.DataFrame([{
            "strike": 100.0, "expiry": past_expiry, "option_type": "call",
            "bid": 2.0, "ask": 2.1,
        }])
        surf = build_iv_surface(quotes, spot=100.0, reference_date=date(2026, 3, 3))
        assert len(surf.points) == 0

    # ── expiries / strikes ────────────────────────────────────────────────────

    def test_expiries_sorted(self):
        surf = self._surface()
        exp = surf.expiries()
        assert exp == sorted(exp)

    def test_strikes_sorted(self):
        surf = self._surface()
        ks = surf.strikes()
        assert ks == sorted(ks)

    def test_expiries_count(self):
        surf = self._surface()
        assert len(surf.expiries()) == 3  # 30, 60, 90 days

    # ── smile_for_expiry ─────────────────────────────────────────────────────

    def test_smile_returns_series(self):
        surf = self._surface()
        expiry = surf.expiries()[0]
        smile = surf.smile_for_expiry(expiry)
        assert isinstance(smile, pd.Series)

    def test_smile_strikes_sorted(self):
        surf = self._surface()
        smile = surf.smile_for_expiry(surf.expiries()[0])
        assert list(smile.index) == sorted(smile.index)

    def test_smile_iv_positive(self):
        surf = self._surface()
        smile = surf.smile_for_expiry(surf.expiries()[0])
        assert (smile > 0).all()

    def test_smile_unknown_expiry_empty(self):
        surf = self._surface()
        smile = surf.smile_for_expiry("9999-12-31")
        assert smile.empty

    def test_smile_has_vol_smile_shape(self):
        """ATM IV should be lower than far OTM IV (smile / smirk)."""
        surf = self._surface()
        smile = surf.smile_for_expiry(surf.expiries()[0])
        if len(smile) >= 3:
            import numpy as np
            strikes_arr = np.array(smile.index, dtype=float)
            atm_idx = int(np.argmin(np.abs(strikes_arr - surf.spot)))
            atm_k = smile.index[atm_idx]
            atm_iv = smile[atm_k]
            # At least one wing should be higher
            wings = smile.drop(atm_k)
            assert wings.max() >= atm_iv

    # ── term_structure ────────────────────────────────────────────────────────

    def test_term_structure_returns_series(self):
        surf = self._surface()
        ts = surf.term_structure()
        assert isinstance(ts, pd.Series)

    def test_term_structure_length_matches_expiries(self):
        surf = self._surface()
        ts = surf.term_structure()
        assert len(ts) == len(surf.expiries())

    def test_term_structure_sorted_by_expiry(self):
        surf = self._surface()
        ts = surf.term_structure()
        assert list(ts.index) == sorted(ts.index)

    def test_term_structure_iv_positive(self):
        surf = self._surface()
        ts = surf.term_structure()
        assert (ts > 0).all()

    def test_term_structure_moneyness(self):
        """Slightly OTM moneyness should still return a series."""
        surf = self._surface()
        ts = surf.term_structure(moneyness=1.05)
        assert not ts.empty

    # ── interpolate ───────────────────────────────────────────────────────────

    def test_interpolate_returns_float(self):
        surf = self._surface()
        iv = surf.interpolate(100.0, 0.25)
        assert iv is not None
        assert isinstance(iv, float)

    def test_interpolate_positive_iv(self):
        surf = self._surface()
        iv = surf.interpolate(100.0, 0.25)
        assert iv > 0

    def test_interpolate_nearest_method(self):
        surf = self._surface()
        iv = surf.interpolate(100.0, 0.25, method="nearest")
        assert iv is not None and iv > 0

    def test_interpolate_empty_surface_returns_none(self):
        empty = IVSurface()
        assert empty.interpolate(100.0, 0.5) is None

    def test_interpolate_iv_wrapper_function(self):
        surf = self._surface()
        iv = interpolate_iv(surf, 100.0, 0.25)
        assert iv is not None and iv > 0

    def test_interpolate_extrapolate_low_strike(self):
        """Strike below min should return nearest (no crash)."""
        surf = self._surface()
        iv = surf.interpolate(1.0, 0.25)  # far below min strike
        assert iv is not None

    def test_interpolate_extrapolate_far_T(self):
        """T beyond max should return nearest (no crash)."""
        surf = self._surface()
        iv = surf.interpolate(100.0, 10.0)  # far beyond max T
        assert iv is not None

    # ── to_dataframe ─────────────────────────────────────────────────────────

    def test_to_dataframe_shape(self):
        surf = self._surface()
        df = surf.to_dataframe()
        assert len(df) == len(surf.points)
        assert set(df.columns) >= {"strike", "expiry", "T", "option_type", "market_mid", "implied_vol"}

    def test_to_dataframe_no_nulls_in_strike(self):
        surf = self._surface()
        df = surf.to_dataframe()
        assert df["strike"].notna().all()

    # ── convenience wrappers ──────────────────────────────────────────────────

    def test_smile_for_expiry_wrapper(self):
        surf = self._surface()
        smile = smile_for_expiry(surf, surf.expiries()[0])
        assert isinstance(smile, pd.Series)

    def test_term_structure_wrapper(self):
        surf = self._surface()
        ts = term_structure(surf)
        assert isinstance(ts, pd.Series)

    # ── internal helpers ──────────────────────────────────────────────────────

    def test_bracket_below_min(self):
        arr = np.array([1.0, 2.0, 3.0])
        lo, hi = _bracket(arr, 0.5)
        assert lo == 0 and hi == 0

    def test_bracket_above_max(self):
        arr = np.array([1.0, 2.0, 3.0])
        lo, hi = _bracket(arr, 5.0)
        assert lo == 2 and hi == 2

    def test_bracket_middle(self):
        arr = np.array([1.0, 2.0, 3.0])
        lo, hi = _bracket(arr, 2.5)
        assert lo == 1 and hi == 2

    def test_build_grid_empty_input(self):
        grid = _build_grid([])
        assert grid.empty

    def test_build_grid_single_point(self):
        pts = [IVPoint(strike=100.0, expiry="2026-06-03", T=0.25,
                       option_type="call", market_mid=5.0, implied_vol=0.20)]
        grid = _build_grid(pts)
        assert not grid.empty
        assert 100.0 in grid.columns

    # ── public __init__ exports ───────────────────────────────────────────────

    def test_imports_from_init(self):
        from phinance.options import (
            black_scholes_call, black_scholes_put, implied_volatility,
            IVSurface, build_iv_surface, interpolate_iv,
            smile_for_expiry, term_structure,
        )
        assert callable(build_iv_surface)

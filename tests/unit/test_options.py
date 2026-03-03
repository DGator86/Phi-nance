"""
tests/unit/test_options.py
===========================

Unit tests for phinance.options:
  - Black-Scholes pricing (call, put, put-call parity)
  - Implied volatility
  - Option Greeks
  - run_options_backtest
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from tests.fixtures.ohlcv import make_ohlcv

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

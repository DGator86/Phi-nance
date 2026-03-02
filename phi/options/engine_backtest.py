"""
Options Engine Backtest
=======================
Walk-forward simulation that drives the *full* Phi-nance options stack:

    OHLCV bar
        ↓
    RegimeEngine.run_latest()   → regime_probs, composite_signal, score
        ↓
    make_synthetic_chain()      → BS-priced chain with realistic IV skew + OI
        ↓
    GammaSurface.compute_features() → gamma_net, gamma_wall_distance, …
        ↓
    OptionsEngine.select_trade()    → OptionsTrade (structure, legs, confidence)
        ↓
    BS mark-to-market daily         → position P&L
        ↓
    Exit on profit target / stop / DTE expiry

Unlike the simple delta-approximation in ``phi.options.backtest``, this module:

* Uses actual Black-Scholes pricing (entry *and* daily MTM) for all legs.
* Supports the full set of L1 / L2 / L3 structures chosen by OptionsEngine.
* Feeds a realistic synthetic chain (IV skew + vega-weighted OI) into
  GammaSurface so the engine gets meaningful gamma-regime signals.
* Returns a rich results dict with per-trade details (structure, level,
  regime at entry, IV regime, confidence) suitable for analysis.

Limitations
-----------
* Historical implied volatility is approximated from 21-day realised vol
  × an IV-premium multiplier (default 1.15).  No actual IV history is used.
* GEX signals derived from the synthetic chain are plausible but not ground-
  truth (no real historical open-interest data).
* Calendar spreads require two DTE levels; the synthetic chain supports this
  via ``dte_short`` / ``dte_long`` parameters.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# BS kernel (self-contained so this module has no circular imports)
# ──────────────────────────────────────────────────────────────────────────────

def _bs(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """Black-Scholes price. Returns 0 on invalid inputs."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(0.0, S - K) if opt == "call" else max(0.0, K - S)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if opt == "call":
            return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
        return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    except Exception:
        return 0.0


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0:
            return 1.0 if opt == "call" else -1.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(norm.cdf(d1) if opt == "call" else -norm.cdf(-d1))
    except Exception:
        return 0.0


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))
    except Exception:
        return 0.0


def _bs_theta(S: float, K: float, T: float, r: float, sigma: float, opt: str) -> float:
    """Daily theta ($)."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        base = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        if opt == "call":
            base -= r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            base += r * K * math.exp(-r * T) * norm.cdf(-d2)
        return float(base / 365)
    except Exception:
        return 0.0


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1% IV change."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return float(S * norm.pdf(d1) * math.sqrt(T) / 100)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic chain builder
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_chain(
    spot: float,
    bar_date: date,
    hist_vol: float,
    dte_short: int = 21,
    dte_long: int = 45,
    n_strikes: int = 25,
    iv_premium: float = 1.15,
    put_skew: float = 0.30,
    r: float = 0.05,
) -> pd.DataFrame:
    """
    Build a synthetic options chain for a single bar using Black-Scholes.

    Parameters
    ----------
    spot        : current underlying price
    bar_date    : calendar date of this bar (sets expiration dates)
    hist_vol    : 21-day realised volatility (annualised, e.g. 0.22)
    dte_short   : front-month DTE (for calendar spreads / covered call)
    dte_long    : main DTE used for all other structures
    n_strikes   : number of strike levels around spot
    iv_premium  : multiplier applied to hist_vol to get ATM IV (VIX > HV effect)
    put_skew    : extra IV applied to OTM puts (mimics left-tail skew)
    r           : risk-free rate

    Returns
    -------
    DataFrame with columns matching the schema expected by GammaSurface
    and OptionsEngine: strike, expiration, optiontype, openinterest,
    impliedvolatility, delta, gamma, theta, vega, bid, ask, last, volume.
    """
    hist_vol = max(hist_vol, 0.05)
    atm_iv   = hist_vol * iv_premium

    strike_range = np.linspace(spot * 0.80, spot * 1.20, n_strikes)

    rows: List[Dict[str, Any]] = []
    for dte in (dte_short, dte_long):
        expiry = (bar_date + timedelta(days=dte)).isoformat()
        T = dte / 365.0

        for K in strike_range:
            moneyness = math.log(K / spot)
            # Put skew: OTM puts have higher IV (negative moneyness)
            skew_adj = put_skew * max(-moneyness, 0.0) * 2.0
            # Call smile: slight OTM call premium
            smile_adj = 0.05 * max(moneyness, 0.0)
            iv = float(np.clip(atm_iv + skew_adj + smile_adj, 0.05, 2.0))

            for opt_type in ("call", "put"):
                price = _bs(spot, K, T, r, iv, opt_type)
                d     = _bs_delta(spot, K, T, r, iv, opt_type)
                g     = _bs_gamma(spot, K, T, r, iv)
                th    = _bs_theta(spot, K, T, r, iv, opt_type)
                vg    = _bs_vega(spot, K, T, r, iv)

                # Vega-weighted OI (realistic: most OI near ATM)
                oi_weight = float(np.exp(-0.5 * (moneyness / 0.08) ** 2))
                oi = max(10, int(oi_weight * 5000))

                spread = max(0.01, price * 0.04)  # 4% bid-ask spread
                bid = max(0.0, price - spread / 2)
                ask = price + spread / 2

                rows.append({
                    "strike":            round(K, 2),
                    "expiration":        expiry,
                    "optiontype":        opt_type,
                    "impliedvolatility": round(iv, 4),
                    "delta":             round(d, 4),
                    "gamma":             round(g, 6),
                    "theta":             round(th, 4),
                    "vega":              round(vg, 4),
                    "bid":               round(bid, 2),
                    "ask":              round(ask, 2),
                    "last":              round(price, 2),
                    "openinterest":      oi,
                    "volume":            max(1, oi // 5),
                })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Position mark-to-market
# ──────────────────────────────────────────────────────────────────────────────

def _mtm_position(
    legs,           # List[OptionsLeg]
    spot: float,
    days_remaining: int,
    r: float,
    current_iv: float,
) -> float:
    """
    Mark a multi-leg position to market.

    Returns current total value per-share (not per-contract).
    Credit strategies use negative entry_cost, so P&L = current_value - entry_cost.
    """
    T = max(days_remaining, 0) / 365.0
    total = 0.0
    for leg in legs:
        price = _bs(spot, leg.strike, T, r, current_iv, leg.option_type)
        sign  = 1.0 if leg.action == "buy" else -1.0
        total += sign * price
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Hist-vol helper
# ──────────────────────────────────────────────────────────────────────────────

def _hist_vol(closes: np.ndarray, window: int = 21) -> float:
    """Annualised realised volatility from recent log-returns."""
    if len(closes) < 3:
        return 0.20
    returns = np.diff(np.log(np.maximum(closes[-window - 1:], 1e-10)))
    if len(returns) < 2:
        return 0.20
    return float(np.std(returns) * math.sqrt(252))


# ──────────────────────────────────────────────────────────────────────────────
# Main backtest entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_engine_backtest(
    ohlcv: pd.DataFrame,
    symbol: str = "SPY",
    initial_capital: float = 100_000.0,
    position_pct: float = 0.10,
    lookback_bars: int = 60,
    min_confidence: float = 0.40,
    dte_days: int = 30,
    dte_short: int = 14,
    exit_profit_pct: float = 0.50,
    exit_stop_pct: float = 0.30,
    iv_premium: float = 1.15,
    put_skew: float = 0.25,
    max_open_positions: int = 1,
    r: float = 0.05,
    progress_cb=None,
) -> dict:
    """
    Walk-forward options engine backtest.

    At each bar the full MFT pipeline runs on the rolling lookback window,
    selects an options structure, enters with BS pricing, and marks to market
    daily until profit target / stop / DTE expiry.

    Parameters
    ----------
    ohlcv             : DataFrame with open/high/low/close/volume, datetime index
    symbol            : ticker label (used for logging only)
    initial_capital   : starting portfolio value
    position_pct      : fraction of capital allocated per trade
    lookback_bars     : bars fed to RegimeEngine (min warm-up window)
    min_confidence    : OptionsEngine confidence gate
    dte_days          : target DTE for main option leg
    dte_short         : DTE for front-month / short leg
    exit_profit_pct   : close position when P&L / max_risk >= this
    exit_stop_pct     : close position when P&L / max_risk <= -this
    iv_premium        : IV = hist_vol × iv_premium (VIX premium effect)
    put_skew          : extra IV on OTM puts (left-tail skew)
    max_open_positions: maximum simultaneous open positions
    r                 : risk-free rate
    progress_cb       : optional callable(fraction: float) for UI progress bars

    Returns
    -------
    dict with:
      portfolio_value  : list[float] — daily portfolio value
      trades           : list[dict] — full trade log
      total_return     : float
      cagr             : float
      max_drawdown     : float
      sharpe           : float
      win_rate         : float
      n_trades         : int
      structures_used  : dict[str, int] — count per structure type
      regimes_at_entry : dict[str, int] — count per dominant regime
    """
    close = ohlcv["close"].values.astype(float)
    n     = len(close)

    if n < lookback_bars + 5:
        return _empty_result(initial_capital)

    # Resolve bar dates
    if hasattr(ohlcv.index, "date"):
        bar_dates = [d.date() if hasattr(d, "date") else date.today() for d in ohlcv.index]
    else:
        bar_dates = [date.today() + timedelta(days=i) for i in range(n)]

    # ── Load engine components (graceful degradation) ───────────────────
    try:
        from regime_engine.scanner import RegimeEngine, load_config
        cfg = load_config()
        regime_engine = RegimeEngine(cfg)
        _have_regime = True
        logger.info("run_engine_backtest: RegimeEngine loaded")
    except Exception as exc:
        logger.warning("RegimeEngine unavailable (%s) — using neutral probs", exc)
        regime_engine = None
        cfg = {}
        _have_regime = False

    try:
        from regime_engine.gamma_surface import GammaSurface
        gamma_surface = GammaSurface(config={
            "kernel_width_pct": 0.005,
            "min_oi": 50,
            "max_dte": max(dte_days, dte_short) + 5,
            "gex_flip_threshold": 0.05,
        })
        _have_gamma = True
    except Exception as exc:
        logger.warning("GammaSurface unavailable: %s", exc)
        gamma_surface = None
        _have_gamma = False

    try:
        from regime_engine.options_engine import OptionsEngine
        options_engine = OptionsEngine(config=cfg.get("options_engine", {}))
        _have_options = True
        logger.info("run_engine_backtest: OptionsEngine loaded")
    except Exception as exc:
        logger.warning("OptionsEngine unavailable (%s) — using AI advisor fallback", exc)
        options_engine = None
        _have_options = False

    # ── AI Advisor fallback (when OptionsEngine unavailable or as supplement)
    try:
        from phi.options.ai_advisor import OptionsAIAdvisor
        ai_advisor = OptionsAIAdvisor(use_ollama=False)  # rules-only for backtest speed
        _have_ai = True
    except Exception as exc:
        logger.warning("OptionsAIAdvisor unavailable: %s", exc)
        ai_advisor = None
        _have_ai = False

    if not _have_options and not _have_ai:
        logger.error("Neither OptionsEngine nor AI advisor available — backtest aborted")
        return _empty_result(initial_capital)

    # ── Walk-forward loop ────────────────────────────────────────────────
    capital      = float(initial_capital)
    pv_series    = [capital] * lookback_bars        # pad warm-up with flat line
    positions: List[Dict[str, Any]] = []             # active positions
    trade_log: List[Dict[str, Any]] = []

    for i in range(lookback_bars, n):
        spot     = float(close[i])
        bar_date = bar_dates[i]
        if progress_cb:
            progress_cb((i - lookback_bars) / max(n - lookback_bars, 1))

        # ── 1. Compute current IV estimate ───────────────────────────────
        hv   = _hist_vol(close[:i + 1])
        cur_iv = hv * iv_premium

        # ── 2. Mark open positions to market & check exits ───────────────
        closed_ids = []
        for pos in positions:
            days_left = pos["dte_days"] - (i - pos["entry_bar"])
            mtm       = _mtm_position(pos["legs"], spot, days_left, r, cur_iv)
            pos_pnl   = mtm - pos["entry_value"]  # per-share net P&L

            # Scale to dollar P&L
            dollar_pnl = pos_pnl * pos["notional"] / max(abs(pos["entry_value"]), 1e-6)
            capital_now = pos["capital_at_entry"] + dollar_pnl
            pos["latest_capital"] = capital_now
            pos["days_held"]       = i - pos["entry_bar"]

            # Exit conditions
            max_risk = max(abs(pos["entry_value"]), 1e-6)
            cum_ret  = pos_pnl / max_risk
            exit_reason = None
            if cum_ret >= exit_profit_pct:
                exit_reason = "profit"
            elif cum_ret <= -exit_stop_pct:
                exit_reason = "stop"
            elif days_left <= 0:
                exit_reason = "expiry"

            if exit_reason:
                trade_log.append({
                    "entry_bar":    pos["entry_bar"],
                    "exit_bar":     i,
                    "entry_date":   str(bar_dates[pos["entry_bar"]]),
                    "exit_date":    str(bar_date),
                    "symbol":       symbol,
                    "structure":    pos["structure"],
                    "level":        pos["level"],
                    "regime":       pos["regime"],
                    "vol_regime":   pos["vol_regime"],
                    "gex_regime":   pos["gex_regime"],
                    "confidence":   round(pos["confidence"], 4),
                    "entry_spot":   round(pos["entry_spot"], 2),
                    "exit_spot":    round(spot, 2),
                    "entry_iv":     round(pos["entry_iv"], 4),
                    "days_held":    i - pos["entry_bar"],
                    "exit_reason":  exit_reason,
                    "pnl_$":        round(dollar_pnl, 2),
                    "cum_ret_%":    round(cum_ret * 100, 2),
                    "n_legs":       len(pos["legs"]),
                })
                capital = capital_now
                closed_ids.append(id(pos))

        positions = [p for p in positions if id(p) not in closed_ids]

        # ── 3. RegimeEngine at this bar ──────────────────────────────────
        if _have_regime:
            try:
                window_df   = ohlcv.iloc[max(0, i - lookback_bars + 1): i + 1]
                ticker_res  = regime_engine.run_latest(window_df)
                regime_probs = dict(getattr(ticker_res, "regime_probs", {}))
                if not regime_probs:
                    # run() returns a full time-series dict; take last row
                    out = regime_engine.run(window_df)
                    rp_df = out.get("regime_probs")
                    if rp_df is not None and not rp_df.empty:
                        regime_probs = rp_df.iloc[-1].to_dict()
            except Exception as exc:
                logger.debug("RegimeEngine bar %d: %s", i, exc)
                regime_probs = {}
        else:
            regime_probs = {"RANGE": 0.5, "TREND_UP": 0.25, "TREND_DN": 0.25}

        # ── 4. Synthetic chain + GammaSurface ────────────────────────────
        chain_df = make_synthetic_chain(
            spot=spot, bar_date=bar_date, hist_vol=hv,
            dte_short=dte_short, dte_long=dte_days,
            iv_premium=iv_premium, put_skew=put_skew, r=r,
        )

        if _have_gamma and gamma_surface is not None:
            try:
                gamma_features = gamma_surface.compute_features(chain_df, spot)
            except Exception:
                gamma_features = _neutral_gamma()
        else:
            gamma_features = _neutral_gamma()

        # ── 5. OptionsEngine: select trade ───────────────────────────────
        trade = None
        if len(positions) < max_open_positions:
            # Try OptionsEngine first
            if _have_options and options_engine is not None:
                try:
                    trade = options_engine.select_trade(
                        regime_probs=regime_probs,
                        gamma_features=gamma_features,
                        chain_df=chain_df,
                        spot=spot,
                        hist_vol_ann=hv,
                        l2_signals=None,
                        min_confidence=min_confidence,
                    )
                except Exception as exc:
                    logger.debug("OptionsEngine bar %d: %s", i, exc)

            # AI Advisor fallback when OptionsEngine returns None or unavailable
            if trade is None and _have_ai and ai_advisor is not None:
                try:
                    # Classify IV regime
                    atm_iv = cur_iv
                    if hv > 0:
                        iv_ratio = atm_iv / hv
                        if iv_ratio >= 1.30:
                            _iv_reg = "HIGH_IV"
                        elif iv_ratio <= 0.80:
                            _iv_reg = "LOW_IV"
                        else:
                            _iv_reg = "NORMAL"
                    else:
                        _iv_reg = "NORMAL"

                    rec = ai_advisor.recommend(
                        symbol=symbol, spot=spot, hist_vol=hv,
                        regime_probs=regime_probs,
                        gamma_features=gamma_features,
                        iv_regime=_iv_reg,
                    )
                    if rec.is_actionable(min_confidence):
                        # Convert AI recommendation into a synthetic OptionsTrade
                        trade = _ai_rec_to_trade(rec, chain_df, spot, cur_iv, dte_days, r)
                except Exception as exc:
                    logger.debug("AI advisor bar %d: %s", i, exc)

        # ── 6. Enter position ────────────────────────────────────────────
        if trade is not None:
            entry_val = sum(
                (1.0 if leg.action == "buy" else -1.0) * leg.mid_price
                for leg in trade.legs
            )
            notional = capital * position_pct

            positions.append({
                "entry_bar":       i,
                "entry_spot":      spot,
                "entry_value":     entry_val,
                "entry_iv":        cur_iv,
                "notional":        notional,
                "capital_at_entry": capital,
                "latest_capital":  capital,
                "dte_days":        dte_days,
                "days_held":       0,
                "legs":            trade.legs,
                "structure":       trade.structure,
                "level":           trade.level,
                "regime":          trade.regime,
                "vol_regime":      trade.vol_regime,
                "gex_regime":      trade.gex_regime,
                "confidence":      trade.confidence,
            })

        # ── 7. Current portfolio value ───────────────────────────────────
        # Sum capital + unrealised P&L from open positions
        unrealised = sum(
            (p["latest_capital"] - p["capital_at_entry"]) for p in positions
        )
        pv_series.append(capital + unrealised)

    # Close any remaining open positions at last price
    if positions:
        last_spot = float(close[-1])
        last_iv   = _hist_vol(close) * iv_premium
        for pos in positions:
            days_left = max(0, pos["dte_days"] - (n - 1 - pos["entry_bar"]))
            mtm       = _mtm_position(pos["legs"], last_spot, days_left, r, last_iv)
            pos_pnl   = mtm - pos["entry_value"]
            dollar_pnl = pos_pnl * pos["notional"] / max(abs(pos["entry_value"]), 1e-6)
            trade_log.append({
                "entry_bar":    pos["entry_bar"],
                "exit_bar":     n - 1,
                "entry_date":   str(bar_dates[pos["entry_bar"]]),
                "exit_date":    str(bar_dates[-1]),
                "symbol":       symbol,
                "structure":    pos["structure"],
                "level":        pos["level"],
                "regime":       pos["regime"],
                "vol_regime":   pos["vol_regime"],
                "gex_regime":   pos["gex_regime"],
                "confidence":   round(pos["confidence"], 4),
                "entry_spot":   round(pos["entry_spot"], 2),
                "exit_spot":    round(last_spot, 2),
                "entry_iv":     round(pos["entry_iv"], 4),
                "days_held":    n - 1 - pos["entry_bar"],
                "exit_reason":  "end_of_data",
                "pnl_$":        round(dollar_pnl, 2),
                "cum_ret_%":    round(
                    (pos_pnl / max(abs(pos["entry_value"]), 1e-6)) * 100, 2
                ),
                "n_legs":       len(pos["legs"]),
            })

    # ── Analytics ─────────────────────────────────────────────────────────
    pv_arr = np.array(pv_series, dtype=float)
    tr = float(pv_arr[-1] / initial_capital - 1) if initial_capital else 0.0
    n_years = max(len(ohlcv) / 252.0, 1.0 / 12)
    cagr = float((1 + tr) ** (1 / n_years) - 1)
    peak = np.maximum.accumulate(pv_arr)
    dd = (pv_arr - peak) / np.where(peak > 0, peak, 1e-8)
    max_dd = float(np.min(dd))
    pv_ret = np.diff(pv_arr) / np.maximum(pv_arr[:-1], 1e-8)
    sharpe = float(np.mean(pv_ret) / np.std(pv_ret) * math.sqrt(252)) \
        if np.std(pv_ret) > 1e-10 else 0.0

    n_trades  = len(trade_log)
    wins      = sum(1 for t in trade_log if t.get("pnl_$", 0) > 0)
    win_rate  = wins / n_trades if n_trades else 0.0

    structures_used: Dict[str, int] = {}
    regimes_at_entry: Dict[str, int] = {}
    for t in trade_log:
        structures_used[t["structure"]]  = structures_used.get(t["structure"], 0) + 1
        regimes_at_entry[t["regime"]]    = regimes_at_entry.get(t["regime"], 0) + 1

    return {
        "portfolio_value":  list(pv_arr),
        "trades":           trade_log,
        "total_return":     tr,
        "cagr":             cagr,
        "max_drawdown":     max_dd,
        "sharpe":           sharpe,
        "win_rate":         win_rate,
        "n_trades":         n_trades,
        "structures_used":  structures_used,
        "regimes_at_entry": regimes_at_entry,
        # Expose engine availability flags for UI diagnostics
        "_regime_engine_ok": _have_regime,
        "_gamma_ok":         _have_gamma,
        "_options_ok":       _have_options,
        "_ai_advisor_ok":    _have_ai,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _neutral_gamma() -> Dict[str, float]:
    return {
        "gamma_wall_distance": 0.0,
        "gamma_net":           0.0,
        "gamma_expiry_days":   30.0,
        "gex_flip_zone":       0.0,
    }


def _ai_rec_to_trade(rec, chain_df, spot, cur_iv, dte_days, r):
    """Convert an OptionsAIAdvisor recommendation into a synthetic OptionsTrade.

    Uses the existing OptionsEngine leg-building machinery when available,
    otherwise constructs simple synthetic legs from the BS chain.
    """
    try:
        from regime_engine.options_engine import OptionsEngine, OptionsLeg, OptionsTrade
    except ImportError:
        return None

    # Build a minimal OptionsEngine to construct legs
    try:
        engine = OptionsEngine(config={})
        chain_norm = engine._normalize_chain(chain_df, spot)
        if chain_norm.empty:
            return None

        legs = engine._build_legs(rec.structure, chain_norm, spot)
        if not legs:
            return None

        # Build the trade object
        return OptionsTrade(
            structure=rec.structure,
            legs=legs,
            level=rec.level,
            regime=rec.regime,
            vol_regime=rec.vol_regime,
            gex_regime=rec.gex_regime,
            confidence=rec.confidence,
            rationale=f"[AI] {rec.reasoning}",
            max_profit=0.0,
            max_loss=0.0,
            breakeven=[],
            net_credit=sum(
                (leg.mid_price if leg.action == "sell" else -leg.mid_price) * 100
                for leg in legs
            ),
        )
    except Exception:
        return None


def _empty_result(initial_capital: float) -> dict:
    return {
        "portfolio_value":  [initial_capital],
        "trades":           [],
        "total_return":     0.0,
        "cagr":             0.0,
        "max_drawdown":     0.0,
        "sharpe":           0.0,
        "win_rate":         0.0,
        "n_trades":         0,
        "structures_used":  {},
        "regimes_at_entry": {},
        "_regime_engine_ok": False,
        "_gamma_ok":         False,
        "_options_ok":       False,
        "_ai_advisor_ok":    False,
    }

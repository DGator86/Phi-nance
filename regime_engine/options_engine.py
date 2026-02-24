"""
Options Engine — Regime-conditioned options trade selection.

Maps the current MFT regime state + volatility regime + GEX features to
an optimal options strategy (L1 directional → L3 non-directional) and
selects specific strikes/expiries from the live options chain.

Volatility Regimes
------------------
  HIGH_IV  — implied volatility > 1.3× realized → sell premium structures
  NORMAL   — IV ≈ HV → balanced
  LOW_IV   — implied volatility < 0.8× realized → buy premium structures

L1 Structures (directional, delta > 0.5)
  long_call        — buy ATM call; regime: TREND_UP + LOW_IV
  long_put         — buy ATM put;  regime: TREND_DN + LOW_IV
  bull_call_spread — buy ATM call, sell OTM call; TREND_UP + NORMAL
  bear_put_spread  — buy ATM put,  sell OTM put;  TREND_DN + NORMAL
  bull_put_spread  — sell OTM put, buy further OTM put; TREND_UP + HIGH_IV
  bear_call_spread — sell OTM call, buy further OTM call; TREND_DN + HIGH_IV

L2 Structures (semi-directional, defined-risk)
  collar       — long stock + long put + short call; HIGHVOL hedge
  covered_call — long stock + short ATM call; RANGE + HIGH_IV

L3 Structures (non-directional, volatility plays)
  long_straddle   — long ATM call + long ATM put; BREAKOUT + LOW_IV
  long_strangle   — long OTM call + long OTM put; BREAKOUT + LOW_IV, wider
  short_straddle  — short ATM call + short ATM put; RANGE + HIGH_IV + PINNING
  iron_condor     — sell OTM strangle, buy wider OTM strangle; RANGE + HIGH_IV
  iron_butterfly  — sell ATM straddle, buy OTM wings; tight RANGE + PINNING
  calendar_spread — sell front-month ATM, buy back-month ATM; term-structure play

GEX Regime Classification
--------------------------
  PINNING  — gamma_net > +0.3 (dealers long gamma, absorb moves → range)
  NEUTRAL  — |gamma_net| <= 0.3
  AMPLIFY  — gamma_net < -0.3 (dealers short gamma, amplify moves → breakout)
  FLIP     — gex_flip_zone == 1 (near zero-crossing → unstable)

Usage
-----
  >>> from regime_engine.options_engine import OptionsEngine
  >>> engine = OptionsEngine(cfg['options_engine'])
  >>> trade = engine.select_trade(
  ...     regime_probs={'TREND_UP': 0.6, 'RANGE': 0.2, ...},
  ...     gamma_features={'gamma_net': 0.1, 'gex_flip_zone': 0, ...},
  ...     chain_df=fetcher.options_chain('AAPL'),
  ...     spot=175.00,
  ...     hist_vol_ann=0.22,
  ... )
  >>> print(trade)
  # OptionsTrade(structure='bull_call_spread', legs=[...], confidence=0.73, ...)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptionsLeg:
    """A single leg of a multi-leg options position."""
    option_type: str          # 'call' or 'put'
    action:      str          # 'buy' or 'sell'
    strike:      float
    expiration:  str          # 'YYYY-MM-DD'
    dte:         int          # days to expiration
    delta:       float = 0.0
    gamma:       float = 0.0
    theta:       float = 0.0
    vega:        float = 0.0
    iv:          float = 0.0
    mid_price:   float = 0.0
    quantity:    int   = 1    # number of contracts (each = 100 shares)


@dataclass
class OptionsTrade:
    """
    A complete options trade recommendation.

    Attributes
    ----------
    structure    : strategy name (e.g. 'iron_condor')
    legs         : ordered list of OptionsLeg objects
    level        : L1 / L2 / L3
    regime       : dominant MFT regime that triggered this trade
    vol_regime   : HIGH_IV / NORMAL / LOW_IV
    gex_regime   : PINNING / NEUTRAL / AMPLIFY / FLIP
    confidence   : composite confidence ∈ [0, 1]
    rationale    : human-readable explanation
    max_profit   : estimated max profit per spread unit
    max_loss     : estimated max loss per spread unit
    breakeven    : list of breakeven price(s)
    net_credit   : positive = credit received; negative = debit paid
    """
    structure:   str
    legs:        List[OptionsLeg]
    level:       str              # 'L1', 'L2', 'L3'
    regime:      str
    vol_regime:  str
    gex_regime:  str
    confidence:  float
    rationale:   str
    max_profit:  float = 0.0
    max_loss:    float = 0.0
    breakeven:   List[float] = field(default_factory=list)
    net_credit:  float = 0.0

    def summary(self) -> str:
        legs_str = ", ".join(
            f"{l.action} {l.quantity}x {l.strike}{l.option_type[0].upper()} {l.expiration}"
            for l in self.legs
        )
        return (
            f"[{self.level}] {self.structure.upper()} | {self.regime} | "
            f"Vol={self.vol_regime} | GEX={self.gex_regime} | "
            f"Conf={self.confidence:.2f} | {legs_str} | "
            f"Max P/L: ${self.max_profit:.0f}/${self.max_loss:.0f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Regime → structure mapping
# ──────────────────────────────────────────────────────────────────────────────

# Keys: (regime, vol_regime, gex_regime) — use 'ANY' as wildcard
# Values: list of (structure_name, level) in priority order
_STRUCTURE_MAP: List[Tuple[Tuple[str, str, str], List[Tuple[str, str]]]] = [
    # ── Trend Up ─────────────────────────────────────────────────────────────
    (('TREND_UP',      'LOW_IV',  'NEUTRAL'), [('long_call',       'L1'), ('bull_call_spread', 'L1')]),
    (('TREND_UP',      'LOW_IV',  'AMPLIFY'), [('long_call',       'L1'), ('long_straddle',    'L3')]),
    (('TREND_UP',      'NORMAL',  'NEUTRAL'), [('bull_call_spread', 'L1'), ('long_call',        'L1')]),
    (('TREND_UP',      'HIGH_IV', 'NEUTRAL'), [('bull_put_spread',  'L1'), ('bull_call_spread', 'L1')]),
    (('TREND_UP',      'HIGH_IV', 'PINNING'), [('bull_put_spread',  'L1'), ('covered_call',     'L2')]),

    # ── Trend Down ────────────────────────────────────────────────────────────
    (('TREND_DN',      'LOW_IV',  'NEUTRAL'), [('long_put',         'L1'), ('bear_put_spread',  'L1')]),
    (('TREND_DN',      'LOW_IV',  'AMPLIFY'), [('long_put',         'L1'), ('long_straddle',    'L3')]),
    (('TREND_DN',      'NORMAL',  'NEUTRAL'), [('bear_put_spread',  'L1'), ('long_put',         'L1')]),
    (('TREND_DN',      'HIGH_IV', 'NEUTRAL'), [('bear_call_spread', 'L1'), ('bear_put_spread',  'L1')]),
    (('TREND_DN',      'HIGH_IV', 'PINNING'), [('bear_call_spread', 'L1'), ('iron_condor',      'L3')]),

    # ── Range ─────────────────────────────────────────────────────────────────
    (('RANGE',         'HIGH_IV', 'PINNING'), [('iron_condor',      'L3'), ('short_straddle',   'L3')]),
    (('RANGE',         'HIGH_IV', 'NEUTRAL'), [('iron_condor',      'L3'), ('covered_call',     'L2')]),
    (('RANGE',         'NORMAL',  'PINNING'), [('iron_condor',      'L3'), ('iron_butterfly',   'L3')]),
    (('RANGE',         'LOW_IV',  'PINNING'), [('iron_butterfly',   'L3'), ('covered_call',     'L2')]),
    (('RANGE',         'LOW_IV',  'NEUTRAL'), [('covered_call',     'L2'), ('bull_put_spread',  'L1')]),

    # ── Breakout ──────────────────────────────────────────────────────────────
    (('BREAKOUT_UP',   'LOW_IV',  'ANY'),     [('long_call',        'L1'), ('long_straddle',    'L3')]),
    (('BREAKOUT_UP',   'NORMAL',  'AMPLIFY'), [('long_straddle',    'L3'), ('long_call',        'L1')]),
    (('BREAKOUT_UP',   'HIGH_IV', 'AMPLIFY'), [('bull_put_spread',  'L1'), ('long_strangle',    'L3')]),
    (('BREAKOUT_DN',   'LOW_IV',  'ANY'),     [('long_put',         'L1'), ('long_straddle',    'L3')]),
    (('BREAKOUT_DN',   'NORMAL',  'AMPLIFY'), [('long_straddle',    'L3'), ('long_put',         'L1')]),
    (('BREAKOUT_DN',   'HIGH_IV', 'AMPLIFY'), [('bear_call_spread', 'L1'), ('long_strangle',    'L3')]),

    # ── Exhaustion / Reversal ─────────────────────────────────────────────────
    (('EXHAUST_REV',   'ANY',     'FLIP'),    [('long_straddle',    'L3'), ('long_strangle',    'L3')]),
    (('EXHAUST_REV',   'ANY',     'ANY'),     [('long_straddle',    'L3'), ('bear_put_spread',  'L1')]),

    # ── Volatility Regimes ────────────────────────────────────────────────────
    (('HIGHVOL',       'HIGH_IV', 'ANY'),     [('long_put',         'L1'), ('collar',           'L2')]),
    (('HIGHVOL',       'NORMAL',  'ANY'),     [('bear_put_spread',  'L1'), ('long_strangle',    'L3')]),
    (('LOWVOL',        'HIGH_IV', 'PINNING'), [('iron_condor',      'L3'), ('iron_butterfly',   'L3')]),
    (('LOWVOL',        'LOW_IV',  'PINNING'), [('iron_butterfly',   'L3'), ('calendar_spread',  'L3')]),

    # ── GEX Flip wildcard ─────────────────────────────────────────────────────
    (('ANY',           'ANY',     'FLIP'),    [('long_straddle',    'L3'), ('long_strangle',    'L3')]),
]

# Level labels for structures not in map
_STRUCTURE_LEVELS: Dict[str, str] = {
    'long_call': 'L1', 'long_put': 'L1',
    'bull_call_spread': 'L1', 'bear_put_spread': 'L1',
    'bull_put_spread': 'L1', 'bear_call_spread': 'L1',
    'collar': 'L2', 'covered_call': 'L2',
    'long_straddle': 'L3', 'long_strangle': 'L3',
    'short_straddle': 'L3', 'iron_condor': 'L3',
    'iron_butterfly': 'L3', 'calendar_spread': 'L3',
}


# ──────────────────────────────────────────────────────────────────────────────
# OptionsEngine
# ──────────────────────────────────────────────────────────────────────────────

class OptionsEngine:
    """
    Regime-conditioned options trade selector.

    Analyzes IV surface vs realized volatility, classifies the GEX regime,
    and selects the optimal options structure + specific legs.

    Parameters
    ----------
    config : dict — 'options_engine' sub-dict from config.yaml
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.target_dte_short  = int(config.get('target_dte_short', 21))   # front-month
        self.target_dte_long   = int(config.get('target_dte_long',  45))   # main expiry
        self.target_dte_back   = int(config.get('target_dte_back',  60))   # back-month (calendar)
        self.iv_high_ratio     = float(config.get('iv_high_ratio',  1.30))  # IV/HV threshold
        self.iv_low_ratio      = float(config.get('iv_low_ratio',   0.80))
        self.gex_pin_thresh    = float(config.get('gex_pin_thresh', 0.30))
        self.gex_amp_thresh    = float(config.get('gex_amp_thresh', -0.30))
        self.spread_width_pct  = float(config.get('spread_width_pct', 0.05))  # OTM spread width
        self.min_oi            = int(config.get('min_oi', 10))
        self.min_iv            = float(config.get('min_iv', 0.05))

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def select_trade(
        self,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
        chain_df: pd.DataFrame,
        spot: float,
        hist_vol_ann: float = 0.20,
        l2_signals: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.35,
    ) -> Optional[OptionsTrade]:
        """
        Select the optimal options trade given current market state.

        Parameters
        ----------
        regime_probs   : dict of {regime_name: probability} from RegimeEngine
        gamma_features : dict from GammaSurface.compute_features()
        chain_df       : options chain DataFrame from fetcher.options_chain()
        spot           : current underlying mid-price
        hist_vol_ann   : annualized realized volatility (e.g. 0.22 = 22%)
        l2_signals     : optional dict from PolygonL2Client.get_snapshot()
        min_confidence : minimum composite confidence to return a trade

        Returns
        -------
        OptionsTrade or None if confidence below threshold / chain empty.
        """
        if chain_df is None or chain_df.empty or spot <= 0:
            return None

        chain = self._normalize_chain(chain_df, spot)
        if chain.empty:
            return None

        vol_regime  = self._classify_vol_regime(chain, spot, hist_vol_ann)
        gex_regime  = self._classify_gex_regime(gamma_features)
        dominant    = self._dominant_regime(regime_probs)
        confidence  = self._compute_confidence(regime_probs, gamma_features, l2_signals)

        if confidence < min_confidence:
            logger.debug(
                "OptionsEngine: confidence %.2f < threshold %.2f, no trade",
                confidence, min_confidence,
            )
            return None

        structure, level = self._lookup_structure(dominant, vol_regime, gex_regime)
        if structure is None:
            logger.debug("OptionsEngine: no structure mapped for %s/%s/%s", dominant, vol_regime, gex_regime)
            return None

        legs = self._build_legs(structure, chain, spot)
        if not legs:
            logger.debug("OptionsEngine: could not build legs for %s", structure)
            return None

        trade = self._assemble_trade(
            structure, level, legs, dominant, vol_regime, gex_regime,
            confidence, regime_probs, gamma_features,
        )
        return trade

    def compute_iv_surface_features(
        self,
        chain_df: pd.DataFrame,
        spot: float,
    ) -> Dict[str, float]:
        """
        Extract IV surface summary features for injection into regime engine.

        Returns
        -------
        dict with:
          iv_atm       — ATM implied volatility (annualized)
          iv_skew      — put/call IV skew (25-delta put IV − call IV)
          iv_term      — IV term structure slope (short/long DTE IV ratio)
          iv_percentile— current ATM IV vs 30-day rolling (0=low, 1=high)
        """
        if chain_df is None or chain_df.empty or spot <= 0:
            return {'iv_atm': 0.0, 'iv_skew': 0.0, 'iv_term': 0.0, 'iv_percentile': 0.5}

        chain = self._normalize_chain(chain_df, spot)
        if chain.empty:
            return {'iv_atm': 0.0, 'iv_skew': 0.0, 'iv_term': 0.0, 'iv_percentile': 0.5}

        return {
            'iv_atm':        self._iv_atm(chain, spot),
            'iv_skew':       self._iv_skew(chain, spot),
            'iv_term':       self._iv_term(chain),
            'iv_percentile': self._iv_percentile(chain, spot),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Regime / vol classification
    # ──────────────────────────────────────────────────────────────────────

    def _classify_vol_regime(
        self,
        chain: pd.DataFrame,
        spot: float,
        hist_vol: float,
    ) -> str:
        atm_iv = self._iv_atm(chain, spot)
        if hist_vol <= 0 or atm_iv <= 0:
            return 'NORMAL'
        ratio = atm_iv / (hist_vol + 1e-10)
        if ratio >= self.iv_high_ratio:
            return 'HIGH_IV'
        if ratio <= self.iv_low_ratio:
            return 'LOW_IV'
        return 'NORMAL'

    def _classify_gex_regime(self, gf: Dict[str, float]) -> str:
        if gf.get('gex_flip_zone', 0) >= 0.5:
            return 'FLIP'
        gn = float(gf.get('gamma_net', 0.0))
        if gn >= self.gex_pin_thresh:
            return 'PINNING'
        if gn <= self.gex_amp_thresh:
            return 'AMPLIFY'
        return 'NEUTRAL'

    @staticmethod
    def _dominant_regime(regime_probs: Dict[str, float]) -> str:
        if not regime_probs:
            return 'RANGE'
        return max(regime_probs, key=lambda k: regime_probs[k])

    def _compute_confidence(
        self,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
        l2_signals: Optional[Dict[str, float]],
    ) -> float:
        """Composite confidence: regime clarity × liquidity quality × GEX stability."""
        # Regime clarity: entropy-based
        probs = np.array(list(regime_probs.values()), dtype=float)
        probs = np.clip(probs, 1e-10, 1.0)
        probs /= probs.sum()
        max_entropy = math.log(len(probs) + 1e-10)
        entropy = -float((probs * np.log(probs)).sum())
        c_regime = float(np.clip(1.0 - entropy / (max_entropy + 1e-10), 0.0, 1.0))

        # GEX stability
        c_gex = 1.0 - float(gamma_features.get('gex_flip_zone', 0.0)) * 0.5

        # Liquidity quality from L2
        c_liq = 1.0
        if l2_signals:
            spread = float(l2_signals.get('spread_bps', 0.0))
            # Wide spread → lower confidence
            c_liq = float(np.clip(1.0 - spread / 50.0, 0.3, 1.0))

        return float(np.clip(c_regime * c_gex * c_liq, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────
    # Structure lookup
    # ──────────────────────────────────────────────────────────────────────

    def _lookup_structure(
        self,
        regime: str,
        vol_regime: str,
        gex_regime: str,
    ) -> Tuple[Optional[str], str]:
        """Return (structure_name, level) from the mapping table."""
        # Try exact match first, then wildcard on gex_regime, then full wildcard
        for key, choices in _STRUCTURE_MAP:
            r, v, g = key
            r_match = (r == 'ANY' or r == regime)
            v_match = (v == 'ANY' or v == vol_regime)
            g_match = (g == 'ANY' or g == gex_regime)
            if r_match and v_match and g_match:
                struct, lvl = choices[0]
                return struct, lvl

        # Fallback: directional default
        if 'TREND_UP' in regime or 'BREAKOUT_UP' in regime:
            return 'long_call', 'L1'
        if 'TREND_DN' in regime or 'BREAKOUT_DN' in regime:
            return 'long_put', 'L1'
        return 'iron_condor', 'L3'

    # ──────────────────────────────────────────────────────────────────────
    # Leg construction
    # ──────────────────────────────────────────────────────────────────────

    def _build_legs(
        self,
        structure: str,
        chain: pd.DataFrame,
        spot: float,
    ) -> List[OptionsLeg]:
        """Dispatch to structure-specific leg builder."""
        builders = {
            'long_call':        self._legs_long_call,
            'long_put':         self._legs_long_put,
            'bull_call_spread':  self._legs_bull_call_spread,
            'bear_put_spread':   self._legs_bear_put_spread,
            'bull_put_spread':   self._legs_bull_put_spread,
            'bear_call_spread':  self._legs_bear_call_spread,
            'long_straddle':     self._legs_long_straddle,
            'long_strangle':     self._legs_long_strangle,
            'short_straddle':    self._legs_short_straddle,
            'iron_condor':       self._legs_iron_condor,
            'iron_butterfly':    self._legs_iron_butterfly,
            'calendar_spread':   self._legs_calendar_spread,
            'covered_call':      self._legs_covered_call,
            'collar':            self._legs_collar,
        }
        fn = builders.get(structure)
        if fn is None:
            return []
        try:
            return fn(chain, spot)
        except Exception as exc:
            logger.warning("OptionsEngine._build_legs(%s) failed: %s", structure, exc)
            return []

    # ── Individual leg builders ────────────────────────────────────────────

    def _legs_long_call(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp = self._pick_expiry(chain, self.target_dte_long)
        c   = self._pick_strike(chain, spot, 'call', exp, delta_target=0.45)
        if c is None:
            return []
        return [self._make_leg(c, 'call', 'buy')]

    def _legs_long_put(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp = self._pick_expiry(chain, self.target_dte_long)
        p   = self._pick_strike(chain, spot, 'put', exp, delta_target=-0.45)
        if p is None:
            return []
        return [self._make_leg(p, 'put', 'buy')]

    def _legs_bull_call_spread(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_long)
        atm  = self._pick_strike(chain, spot, 'call', exp, delta_target=0.50)
        otm  = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + self.spread_width_pct)
        if atm is None or otm is None:
            return []
        return [self._make_leg(atm, 'call', 'buy'), self._make_leg(otm, 'call', 'sell')]

    def _legs_bear_put_spread(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_long)
        atm  = self._pick_strike(chain, spot, 'put', exp, delta_target=-0.50)
        otm  = self._pick_strike(chain, spot, 'put', exp, target_moneyness=1.0 - self.spread_width_pct)
        if atm is None or otm is None:
            return []
        return [self._make_leg(atm, 'put', 'buy'), self._make_leg(otm, 'put', 'sell')]

    def _legs_bull_put_spread(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_short)
        atm  = self._pick_strike(chain, spot, 'put', exp, delta_target=-0.30)
        otm  = self._pick_strike(chain, spot, 'put', exp, target_moneyness=1.0 - self.spread_width_pct)
        if atm is None or otm is None:
            return []
        return [self._make_leg(atm, 'put', 'sell'), self._make_leg(otm, 'put', 'buy')]

    def _legs_bear_call_spread(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_short)
        atm  = self._pick_strike(chain, spot, 'call', exp, delta_target=0.30)
        otm  = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + self.spread_width_pct)
        if atm is None or otm is None:
            return []
        return [self._make_leg(atm, 'call', 'sell'), self._make_leg(otm, 'call', 'buy')]

    def _legs_long_straddle(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp = self._pick_expiry(chain, self.target_dte_long)
        c   = self._pick_strike(chain, spot, 'call', exp, delta_target=0.50)
        p   = self._pick_strike(chain, spot, 'put',  exp, delta_target=-0.50)
        if c is None or p is None:
            return []
        return [self._make_leg(c, 'call', 'buy'), self._make_leg(p, 'put', 'buy')]

    def _legs_long_strangle(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp = self._pick_expiry(chain, self.target_dte_long)
        w   = self.spread_width_pct * 2
        c   = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + w)
        p   = self._pick_strike(chain, spot, 'put',  exp, target_moneyness=1.0 - w)
        if c is None or p is None:
            return []
        return [self._make_leg(c, 'call', 'buy'), self._make_leg(p, 'put', 'buy')]

    def _legs_short_straddle(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp = self._pick_expiry(chain, self.target_dte_short)
        c   = self._pick_strike(chain, spot, 'call', exp, delta_target=0.50)
        p   = self._pick_strike(chain, spot, 'put',  exp, delta_target=-0.50)
        if c is None or p is None:
            return []
        return [self._make_leg(c, 'call', 'sell'), self._make_leg(p, 'put', 'sell')]

    def _legs_iron_condor(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_short)
        w    = self.spread_width_pct
        sc   = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + w)
        lc   = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + w * 2)
        sp   = self._pick_strike(chain, spot, 'put',  exp, target_moneyness=1.0 - w)
        lp   = self._pick_strike(chain, spot, 'put',  exp, target_moneyness=1.0 - w * 2)
        if any(x is None for x in [sc, lc, sp, lp]):
            return []
        return [
            self._make_leg(sp, 'put',  'buy'),
            self._make_leg(sc, 'put',  'sell'),
            self._make_leg(sc, 'call', 'sell'),
            self._make_leg(lc, 'call', 'buy'),
        ]

    def _legs_iron_butterfly(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_short)
        w    = self.spread_width_pct
        atm_c = self._pick_strike(chain, spot, 'call', exp, delta_target=0.50)
        atm_p = self._pick_strike(chain, spot, 'put',  exp, delta_target=-0.50)
        wing_c = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + w)
        wing_p = self._pick_strike(chain, spot, 'put',  exp, target_moneyness=1.0 - w)
        if any(x is None for x in [atm_c, atm_p, wing_c, wing_p]):
            return []
        return [
            self._make_leg(wing_p, 'put',  'buy'),
            self._make_leg(atm_p,  'put',  'sell'),
            self._make_leg(atm_c,  'call', 'sell'),
            self._make_leg(wing_c, 'call', 'buy'),
        ]

    def _legs_calendar_spread(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        front_exp = self._pick_expiry(chain, self.target_dte_short)
        back_exp  = self._pick_expiry(chain, self.target_dte_back)
        if front_exp == back_exp:
            return []
        front = self._pick_strike(chain, spot, 'call', front_exp, delta_target=0.50)
        back  = self._pick_strike(chain, spot, 'call', back_exp,  delta_target=0.50)
        if front is None or back is None:
            return []
        return [self._make_leg(front, 'call', 'sell'), self._make_leg(back, 'call', 'buy')]

    def _legs_covered_call(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp = self._pick_expiry(chain, self.target_dte_short)
        c   = self._pick_strike(chain, spot, 'call', exp, delta_target=0.30)
        if c is None:
            return []
        return [self._make_leg(c, 'call', 'sell')]

    def _legs_collar(self, chain: pd.DataFrame, spot: float) -> List[OptionsLeg]:
        exp  = self._pick_expiry(chain, self.target_dte_long)
        p    = self._pick_strike(chain, spot, 'put',  exp, target_moneyness=1.0 - self.spread_width_pct)
        c    = self._pick_strike(chain, spot, 'call', exp, target_moneyness=1.0 + self.spread_width_pct)
        legs = []
        if p is not None:
            legs.append(self._make_leg(p, 'put', 'buy'))
        if c is not None:
            legs.append(self._make_leg(c, 'call', 'sell'))
        return legs

    # ──────────────────────────────────────────────────────────────────────
    # Strike / expiry selection helpers
    # ──────────────────────────────────────────────────────────────────────

    def _pick_expiry(self, chain: pd.DataFrame, target_dte: int) -> str:
        """Choose the available expiry closest to target_dte."""
        if '_dte' not in chain.columns:
            return ''
        available = chain['_dte'].unique()
        if len(available) == 0:
            return ''
        best_dte = min(available, key=lambda d: abs(d - target_dte))
        rows = chain[chain['_dte'] == best_dte]
        return str(rows['expiration'].iloc[0]) if not rows.empty else ''

    def _pick_strike(
        self,
        chain: pd.DataFrame,
        spot: float,
        option_type: str,
        expiry: str,
        delta_target: Optional[float] = None,
        target_moneyness: Optional[float] = None,
    ) -> Optional[pd.Series]:
        """Pick the best strike row for a given option type/expiry."""
        sub = chain[
            (chain['optiontype'] == option_type) &
            (chain['expiration'] == expiry) &
            (chain['_oi'] >= self.min_oi)
        ].copy()

        if sub.empty:
            return None

        if target_moneyness is not None:
            target_strike = spot * target_moneyness
            sub['_dist'] = (sub['strike'] - target_strike).abs()
        elif delta_target is not None and 'delta' in sub.columns:
            sub['_dist'] = (sub['delta'] - delta_target).abs()
        else:
            sub['_dist'] = (sub['strike'] - spot).abs()

        return sub.loc[sub['_dist'].idxmin()]

    def _make_leg(self, row: pd.Series, option_type: str, action: str) -> OptionsLeg:
        mid = 0.0
        if 'bid' in row.index and 'ask' in row.index:
            b = float(row.get('bid', 0) or 0)
            a = float(row.get('ask', 0) or 0)
            mid = (b + a) / 2.0 if a > b else float(row.get('last', 0) or 0)

        return OptionsLeg(
            option_type = option_type,
            action      = action,
            strike      = float(row.get('strike', 0)),
            expiration  = str(row.get('expiration', '')),
            dte         = int(row.get('_dte', 0)),
            delta       = float(row.get('delta', 0) or 0),
            gamma       = float(row.get('gamma', 0) or 0),
            theta       = float(row.get('theta', 0) or 0),
            vega        = float(row.get('vega', 0) or 0),
            iv          = float(row.get('impliedvolatility', 0) or 0),
            mid_price   = mid,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Trade assembly & P/L estimation
    # ──────────────────────────────────────────────────────────────────────

    def _assemble_trade(
        self,
        structure:    str,
        level:        str,
        legs:         List[OptionsLeg],
        regime:       str,
        vol_regime:   str,
        gex_regime:   str,
        confidence:   float,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
    ) -> OptionsTrade:
        # Net credit/debit: sell legs add premium, buy legs subtract
        net_credit = sum(
            (leg.mid_price if leg.action == 'sell' else -leg.mid_price) * 100
            for leg in legs
        )

        max_profit, max_loss, breakeven = self._estimate_pl(structure, legs)

        rationale = self._make_rationale(structure, regime, vol_regime, gex_regime, regime_probs, gamma_features)

        return OptionsTrade(
            structure   = structure,
            legs        = legs,
            level       = level,
            regime      = regime,
            vol_regime  = vol_regime,
            gex_regime  = gex_regime,
            confidence  = confidence,
            rationale   = rationale,
            max_profit  = max_profit,
            max_loss    = max_loss,
            breakeven   = breakeven,
            net_credit  = net_credit,
        )

    def _estimate_pl(
        self,
        structure: str,
        legs: List[OptionsLeg],
    ) -> Tuple[float, float, List[float]]:
        """Simple max P/L estimation per spread unit (×100 shares)."""
        if not legs:
            return 0.0, 0.0, []

        total_debit  = sum(l.mid_price for l in legs if l.action == 'buy')  * 100
        total_credit = sum(l.mid_price for l in legs if l.action == 'sell') * 100
        net = total_credit - total_debit

        if structure in ('long_call', 'long_put'):
            max_loss   = total_debit
            max_profit = float('inf')
            strikes    = [l.strike for l in legs]
            be = [strikes[0] + total_debit / 100] if structure == 'long_call' else [strikes[0] - total_debit / 100]
            return max_profit, max_loss, be

        if structure in ('bull_call_spread', 'bear_put_spread'):
            strikes  = sorted([l.strike for l in legs])
            width    = (strikes[-1] - strikes[0]) * 100
            max_loss = total_debit
            max_profit = width - total_debit
            be = [strikes[0] + total_debit / 100]
            return max_profit, max_loss, be

        if structure in ('bull_put_spread', 'bear_call_spread'):
            strikes  = sorted([l.strike for l in legs])
            width    = (strikes[-1] - strikes[0]) * 100
            max_profit = total_credit
            max_loss   = width - total_credit
            be = [strikes[0] - total_credit / 100] if structure == 'bull_put_spread' else [strikes[-1] + total_credit / 100]
            return max_profit, max_loss, be

        if structure in ('long_straddle', 'long_strangle'):
            max_loss   = total_debit
            max_profit = float('inf')
            strikes    = sorted([l.strike for l in legs])
            be = [strikes[0] - total_debit / 100, strikes[-1] + total_debit / 100]
            return max_profit, max_loss, be

        if structure in ('short_straddle',):
            max_profit = total_credit
            max_loss   = float('inf')
            strikes    = [l.strike for l in legs if l.option_type == 'call']
            be = []
            if strikes:
                be = [strikes[0] - total_credit / 100, strikes[0] + total_credit / 100]
            return max_profit, max_loss, be

        if structure in ('iron_condor', 'iron_butterfly'):
            buy_strikes  = sorted([l.strike for l in legs if l.action == 'buy'])
            sell_strikes = sorted([l.strike for l in legs if l.action == 'sell'])
            if len(buy_strikes) >= 2 and len(sell_strikes) >= 2:
                wing_width = (sell_strikes[-1] - buy_strikes[0]) * 100
                max_profit = net
                max_loss   = wing_width - net
            else:
                max_profit = net
                max_loss   = 0.0
            return max_profit, max_loss, []

        # Default
        return max(net, 0.0), abs(min(net, 0.0)), []

    @staticmethod
    def _make_rationale(
        structure: str,
        regime: str,
        vol_regime: str,
        gex_regime: str,
        regime_probs: Dict[str, float],
        gamma_features: Dict[str, float],
    ) -> str:
        top_prob = regime_probs.get(regime, 0.0)
        gn = gamma_features.get('gamma_net', 0.0)
        wd = gamma_features.get('gamma_wall_distance', 0.0)
        flip = gamma_features.get('gex_flip_zone', 0)
        parts = [
            f"Regime={regime} ({top_prob:.0%})",
            f"Vol={vol_regime}",
            f"GEX={gex_regime} (net={gn:+.2f}, wall_dist={wd:+.3f})",
        ]
        if flip:
            parts.append("⚠ GEX flip zone — elevated instability")
        parts.append(f"→ {structure}")
        return " | ".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # IV surface helpers
    # ──────────────────────────────────────────────────────────────────────

    def _iv_atm(self, chain: pd.DataFrame, spot: float) -> float:
        """Mean IV of near-ATM options (moneyness 0.97–1.03) in nearest expiry."""
        exp = self._pick_expiry(chain, self.target_dte_long)
        sub = chain[chain['expiration'] == exp].copy() if exp else chain.copy()
        sub = sub[
            (sub['strike'] >= spot * 0.97) &
            (sub['strike'] <= spot * 1.03) &
            (sub['impliedvolatility'] >= self.min_iv)
        ]
        if sub.empty:
            return 0.0
        return float(sub['impliedvolatility'].mean())

    def _iv_skew(self, chain: pd.DataFrame, spot: float) -> float:
        """25-delta put IV minus 25-delta call IV (positive = put skew)."""
        exp = self._pick_expiry(chain, self.target_dte_long)
        sub = chain[chain['expiration'] == exp] if exp else chain
        puts  = sub[sub['optiontype'] == 'put'].copy()
        calls = sub[sub['optiontype'] == 'call'].copy()
        if puts.empty or calls.empty or 'delta' not in sub.columns:
            return 0.0
        puts['_dd']  = (puts['delta']  - (-0.25)).abs()
        calls['_dd'] = (calls['delta'] - 0.25).abs()
        p25_iv = float(puts.loc[puts['_dd'].idxmin(), 'impliedvolatility'])
        c25_iv = float(calls.loc[calls['_dd'].idxmin(), 'impliedvolatility'])
        return p25_iv - c25_iv

    def _iv_term(self, chain: pd.DataFrame) -> float:
        """Short-term / long-term IV ratio (>1 = contango, <1 = backwardation)."""
        exps = sorted(chain['_dte'].unique())
        if len(exps) < 2:
            return 1.0
        short_exp = exps[0]
        long_exp  = exps[-1]
        calls = chain[chain['optiontype'] == 'call']
        iv_short = calls[calls['_dte'] == short_exp]['impliedvolatility'].mean()
        iv_long  = calls[calls['_dte'] == long_exp]['impliedvolatility'].mean()
        if iv_long <= 0:
            return 1.0
        return float(np.clip(iv_short / (iv_long + 1e-10), 0.5, 2.0))

    def _iv_percentile(self, chain: pd.DataFrame, spot: float) -> float:
        """Rough IV percentile: ATM IV vs range of all IVs in chain (0=low, 1=high)."""
        ivs = chain[chain['impliedvolatility'] >= self.min_iv]['impliedvolatility']
        if ivs.empty:
            return 0.5
        atm_iv = self._iv_atm(chain, spot)
        lo, hi = float(ivs.min()), float(ivs.max())
        if hi <= lo:
            return 0.5
        return float(np.clip((atm_iv - lo) / (hi - lo + 1e-10), 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────
    # Chain normalization
    # ──────────────────────────────────────────────────────────────────────

    def _normalize_chain(self, chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
        """Normalize column names, add _dte, _oi, filter bad rows."""
        df = chain_df.copy()
        df.columns = [str(c).lower().replace(' ', '').replace('_', '') for c in df.columns]

        # Resolve columns
        col_map = {
            'strike':            self._find_col(df, ['strike']),
            'expiration':        self._find_col(df, ['expiration', 'expiry', 'expirationdate']),
            'optiontype':        self._find_col(df, ['optiontype', 'type', 'cpflag']),
            'openinterest':      self._find_col(df, ['openinterest', 'oi']),
            'impliedvolatility': self._find_col(df, ['impliedvolatility', 'iv', 'impliedvol']),
            'delta':             self._find_col(df, ['delta']),
            'gamma':             self._find_col(df, ['gamma']),
            'theta':             self._find_col(df, ['theta']),
            'vega':              self._find_col(df, ['vega']),
            'bid':               self._find_col(df, ['bid']),
            'ask':               self._find_col(df, ['ask']),
            'last':              self._find_col(df, ['last', 'lastprice', 'close']),
        }

        # Build normalized df
        out: Dict[str, Any] = {}
        for target, src in col_map.items():
            if src is not None:
                out[target] = pd.to_numeric(df[src], errors='coerce') \
                    if target not in ('expiration', 'optiontype') else df[src]
            else:
                out[target] = 0.0 if target not in ('expiration', 'optiontype') else ''

        result = pd.DataFrame(out)

        # Normalize option type
        result['optiontype'] = (
            result['optiontype'].astype(str).str.lower().str.strip()
            .replace({'c': 'call', 'p': 'put', '': 'call'})
        )

        # Compute DTE
        today = date.today()
        def _dte(s: str) -> int:
            try:
                return max(0, (pd.Timestamp(s).date() - today).days)
            except Exception:
                return 999
        result['_dte'] = result['expiration'].astype(str).map(_dte)
        result['_oi']  = pd.to_numeric(result['openinterest'], errors='coerce').fillna(0)

        # Filter
        result = result[
            result['strike'].notna() &
            (result['strike'] > 0) &
            (result['_dte'] <= 180) &
            (result['_oi'] >= 0)
        ]
        return result.reset_index(drop=True)

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

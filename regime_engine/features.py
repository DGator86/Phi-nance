"""
Feature Engine — Vectorized 1-minute OHLCV feature computation.

All features are computed purely from OHLCV.  No downstream indicators
(RSI, MACD, MA, etc.) are used here — those belong in indicator_library.py.

Output: pd.DataFrame with robust-normalized feature columns ready for the
        Taxonomy Energy Model (tanh will be applied inside taxonomy_engine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# Utility: robust z-score normalization
# ──────────────────────────────────────────────────────────────────────────────

def _robust_zscore(
    series: pd.Series,
    window: int,
    mad_floor: float = 1e-8,
    mad_small_mult: float = 10.0,
    ewma_span: int = 30,
) -> pd.Series:
    """
    Robust z-score using rolling median / MAD.
    Falls back to EWMA std when MAD is too small (flat/degenerate windows).
    """
    roll_median = series.rolling(window, min_periods=max(window // 3, 5)).median()
    roll_mad = (
        (series - roll_median)
        .abs()
        .rolling(window, min_periods=max(window // 3, 5))
        .median()
    )
    roll_mad_clamped = roll_mad.clip(lower=mad_floor)

    ewma_std = series.ewm(span=ewma_span, adjust=False).std().clip(lower=mad_floor)

    # Where MAD is suspiciously small, fall back to EWMA std
    use_ewma = roll_mad < (mad_floor * mad_small_mult)
    effective_std = np.where(use_ewma, ewma_std, roll_mad_clamped * 1.4826)
    effective_std = np.maximum(effective_std, mad_floor)

    return (series - roll_median) / effective_std


# ──────────────────────────────────────────────────────────────────────────────
# Rolling autocorrelation (vectorized via strided approach)
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_autocorr(series: pd.Series, window: int, lag: int) -> pd.Series:
    """Rolling Pearson autocorrelation at given lag using a rolling apply."""

    def _acf(x: np.ndarray) -> float:
        if len(x) <= lag:
            return np.nan
        x1 = x[:-lag]
        x2 = x[lag:]
        if x1.std() < 1e-10 or x2.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(x1, x2)[0, 1])

    return series.rolling(window, min_periods=lag + 5).apply(_acf, raw=True)


# ──────────────────────────────────────────────────────────────────────────────
# Rolling Shannon entropy of return distribution
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    def _entropy(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return 0.0
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return 0.0
        counts, _ = np.histogram(x, bins=bins, range=(x_min, x_max))
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log(probs + 1e-15)))

    return series.rolling(window, min_periods=bins).apply(_entropy, raw=True)


# ──────────────────────────────────────────────────────────────────────────────
# Efficiency Ratio
# ──────────────────────────────────────────────────────────────────────────────

def _efficiency_ratio(log_ret: pd.Series, window: int) -> pd.Series:
    """Kaufman Efficiency Ratio on log returns.

    ER = |cumulative log return over window| / sum(|bar log returns|)
    Range [0, 1]: 1 = perfectly directional, 0 = random walk.
    """
    net = log_ret.rolling(window, min_periods=window // 2).sum().abs()
    path = log_ret.abs().rolling(window, min_periods=window // 2).sum()
    return net / (path + 1e-10)


# ──────────────────────────────────────────────────────────────────────────────
# Main Feature Engine
# ──────────────────────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Computes all regime-inference features from 1-minute OHLCV.

    Parameters
    ----------
    config : dict
        The 'features' sub-dict from config.yaml.

    Usage
    -----
    >>> engine = FeatureEngine(cfg['features'])
    >>> features_df = engine.compute(ohlcv_df)
    """

    # Canonical feature column names (used downstream)
    FEATURE_COLS = [
        "log_return",
        "drift_tscore_15", "drift_tscore_30", "drift_tscore_60",
        "rv_30", "rv_120", "rv_delta",
        "true_range", "range_expansion",
        "er_30", "er_60", "er_120",
        "autocorr",
        "flip_rate",
        "median_cross",
        "impulse_revert",
        "illiquidity",
        "volume_zscore",
        "gap_score",
        "entropy",
        # ── Market Field Theory L2-proxy features ──────────────────────
        # L2: rate of change of price impact (your core formula)
        "d_lambda",
        # L1: effective market mass = 1/lambda (deep book = high mass)
        "mass",
        # L2: mass collapse rate — pre-acceleration signal
        "d_mass_dt",
        # L1 proxy: signed volume pressure = sign(close-open) × volume
        "ofi_proxy",
        # L2 proxy: volume per unit range (high = absorption)
        "absorption_score",
        # L3 proxy: field non-conservativeness (1=trending, -1=mean-reverting)
        "dissipation_proxy",
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features and return a robust-normalized DataFrame.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Must have columns: open, high, low, close, volume.
            Index should be a DatetimeIndex (or integer).

        Returns
        -------
        pd.DataFrame
            Same index as ohlcv, columns = FEATURE_COLS.
            Values are robust z-scores (NaN for warmup rows).
        """
        ohlcv = ohlcv.copy()
        ohlcv.columns = [c.lower() for c in ohlcv.columns]

        open_  = ohlcv["open"].astype(float)
        high   = ohlcv["high"].astype(float)
        low    = ohlcv["low"].astype(float)
        close  = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float).clip(lower=1.0)

        raw = self._compute_raw(open_, high, low, close, volume)
        return self._normalize(raw)

    # ------------------------------------------------------------------
    # Raw (un-normalized) feature computation
    # ------------------------------------------------------------------

    def _compute_raw(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        cfg = self.cfg
        prev_close = close.shift(1)
        log_ret = np.log(close / prev_close)  # NaN at bar 0

        out: Dict[str, pd.Series] = {}

        # ── Log return ─────────────────────────────────────────────────
        out["log_return"] = log_ret

        # ── Rolling drift t-scores ─────────────────────────────────────
        for w in cfg["drift_windows"]:
            roll_mean = log_ret.rolling(w, min_periods=w // 2).mean()
            roll_std  = log_ret.rolling(w, min_periods=w // 2).std()
            out[f"drift_tscore_{w}"] = (
                roll_mean * np.sqrt(w) / (roll_std + 1e-10)
            )

        # ── Realized volatility ────────────────────────────────────────
        for w in cfg["vol_windows"]:
            out[f"rv_{w}"] = log_ret.rolling(w, min_periods=w // 2).std()

        # ── Delta RV ───────────────────────────────────────────────────
        out["rv_delta"] = out["rv_30"] - out["rv_120"]

        # ── True range ─────────────────────────────────────────────────
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["true_range"] = tr

        # ── Range expansion ratio ──────────────────────────────────────
        rexp_w = cfg["range_expansion_window"]
        tr_ma  = tr.rolling(rexp_w, min_periods=rexp_w // 2).mean()
        out["range_expansion"] = tr / (tr_ma + 1e-10)

        # ── Efficiency ratio ───────────────────────────────────────────
        for w in cfg["er_windows"]:
            out[f"er_{w}"] = _efficiency_ratio(log_ret, w)

        # ── Return autocorrelation ─────────────────────────────────────
        out["autocorr"] = _rolling_autocorr(
            log_ret,
            window=cfg["autocorr_window"],
            lag=cfg["autocorr_lag"],
        )

        # ── Flip rate (sign-change frequency) ─────────────────────────
        flip_w = cfg["flip_rate_window"]
        sign_change = (np.sign(log_ret) != np.sign(log_ret.shift(1))).astype(float)
        out["flip_rate"] = sign_change.rolling(flip_w, min_periods=flip_w // 2).mean()

        # ── Median cross frequency ─────────────────────────────────────
        mc_w = cfg["median_cross_window"]
        roll_med = close.rolling(mc_w, min_periods=mc_w // 2).median()
        above = (close > roll_med).astype(float)
        cross = (above != above.shift(1)).astype(float)
        out["median_cross"] = cross.rolling(mc_w, min_periods=mc_w // 2).mean()

        # ── Impulse-then-revert ────────────────────────────────────────
        ir_w   = cfg["impulse_revert_window"]
        ir_sig = cfg["impulse_revert_sigma"]
        roll_std = log_ret.rolling(ir_w, min_periods=ir_w // 2).std()
        large_move   = (log_ret.abs() > roll_std * ir_sig).astype(float)
        opp_sign     = (np.sign(log_ret) != np.sign(log_ret.shift(1))).astype(float)
        impulse_flag = large_move.shift(1) * opp_sign
        out["impulse_revert"] = impulse_flag.rolling(
            ir_w, min_periods=ir_w // 2
        ).mean()

        # ── Illiquidity proxy  |r| / dollar_volume ────────────────────
        dollar_vol = close * volume
        illiq_raw  = log_ret.abs() / (dollar_vol + 1.0)
        ilq_w = cfg["illiquidity_window"]
        out["illiquidity"] = illiq_raw.rolling(ilq_w, min_periods=ilq_w // 2).mean()

        # ── Volume z-score ─────────────────────────────────────────────
        vz_w = cfg["volume_zscore_window"]
        vol_med = volume.rolling(vz_w, min_periods=vz_w // 2).median()
        vol_mad = (
            (volume - vol_med)
            .abs()
            .rolling(vz_w, min_periods=vz_w // 2)
            .median()
        ).clip(lower=1e-6)
        out["volume_zscore"] = (volume - vol_med) / (vol_mad * 1.4826)

        # ── Gap score (open vs prev close) ────────────────────────────
        out["gap_score"] = np.log(open_ / prev_close)

        # ── Shannon entropy of recent return distribution ──────────────
        out["entropy"] = _rolling_entropy(
            log_ret,
            window=cfg["entropy_window"],
            bins=cfg["entropy_bins"],
        )

        # ── Market Field Theory — L2 Proxy Features ────────────────────
        # d_lambda: bar-to-bar change in price impact (rate that book thins)
        # Sign flip at S/R → earliest evaporation warning
        out["d_lambda"] = illiq_raw.diff()

        # mass: effective market mass (inverse price impact)
        # High → deep book, absorptive.  Low → thin book, breakout-prone.
        mass_raw = 1.0 / (illiq_raw + 1e-10)
        out["mass"] = mass_raw

        # d_mass_dt: rate of mass change
        # Strongly negative → market losing resistance without visible price move
        # This is the "variable-mass" term in d²P/dT² = F/M - (dM/dT/M)×dP/dT
        out["d_mass_dt"] = mass_raw.diff()

        # ofi_proxy: OHLCV approximation of Order Flow Imbalance
        # sign(close-open) × volume — positive = buyer aggression dominated bar
        ofi_raw = np.sign(close - open_) * volume
        out["ofi_proxy"] = ofi_raw

        # absorption_score: volume per unit of true range
        # High → lots of volume absorbed per unit of price movement → wall holding
        # Low  → vacuum — price moves easily → breakout fuel
        out["absorption_score"] = volume / (tr + 1e-10)

        # dissipation_proxy: field non-conservativeness proxy
        # Rolling mean of sign(ofi) × sign(ret)
        # +1 → OFI and price perfectly aligned → trending (dissipative field)
        # -1 → OFI opposes price → absorption (conservative/mean-reverting)
        d_w = int(cfg.get("dissipation_window", 30))
        ofi_sign = np.sign(ofi_raw)
        ret_sign = np.sign(log_ret)
        out["dissipation_proxy"] = (
            (ofi_sign * ret_sign)
            .rolling(d_w, min_periods=d_w // 2)
            .mean()
        )

        return pd.DataFrame(out, index=close.index)

    # ------------------------------------------------------------------
    # Robust normalization
    # ------------------------------------------------------------------

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        norm_w      = cfg["robust_norm_window"]
        mad_floor   = cfg["mad_floor"]
        mad_small   = cfg["mad_small_multiplier"]
        ewma_span   = cfg["ewma_norm_span"]

        out = pd.DataFrame(index=raw.index)
        for col in self.FEATURE_COLS:
            if col not in raw.columns:
                out[col] = np.nan
                continue
            out[col] = _robust_zscore(
                raw[col],
                window=norm_w,
                mad_floor=mad_floor,
                mad_small_mult=mad_small,
                ewma_span=ewma_span,
            )
        return out

    # ------------------------------------------------------------------
    # Multi-timeframe aggregation helpers (5m, 15m derived from 1m)
    # ------------------------------------------------------------------

    @staticmethod
    def resample_ohlcv(ohlcv_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Aggregate 1-minute OHLCV to a higher timeframe without external feeds.
        freq examples: '5T', '15T'.
        Requires a DatetimeIndex.
        """
        agg = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }
        df = ohlcv_1m.copy()
        df.columns = [c.lower() for c in df.columns]
        return df.resample(freq).agg(agg).dropna(how="all")

    @staticmethod
    def align_mtf(
        ohlcv_1m: pd.DataFrame,
        features_5m: pd.DataFrame,
        features_15m: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Forward-fill higher-timeframe features onto the 1m index.
        Avoids look-ahead bias by using the *last closed* HTF bar.
        """
        f5  = features_5m.reindex(ohlcv_1m.index, method="ffill")
        f15 = features_15m.reindex(ohlcv_1m.index, method="ffill")
        f5.columns  = [f"tf5_{c}"  for c in f5.columns]
        f15.columns = [f"tf15_{c}" for c in f15.columns]
        return pd.concat([f5, f15], axis=1)

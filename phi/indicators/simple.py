"""
Simple indicator signal computation from OHLCV.
Returns normalized signal series (-1 to 1 scale) for blending.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from phi.indicators.orderflow import (
    compute_cumulative_delta_signal,
    compute_liquidity_signal,
    compute_volume_profile_signal,
    compute_vwap_signal,
    get_order_flow_provider,
)
from phi.logging import get_logger
from phi.mft.complex import complex_potential
from phi.mft.fourier import rolling_spectral_power
from phi.mft.signals import mft_energy_signal, mft_signal
from phi.mft.volume_field import volume_price_interaction

logger = get_logger(__name__)

from phi.indicators.information import (
    compute_entropy_signal,
    compute_fisher_information_signal,
    compute_kld_signal,
    compute_mutual_info_signal,
)
from phi.indicators.information_flow import rolling_granger_causality, rolling_transfer_entropy


def _normalize_signal(s: pd.Series) -> pd.Series:
    """Clip and scale to roughly [-1, 1]."""
    if s.isna().all():
        return s
    q = s.quantile([0.01, 0.99])
    lo, hi = q.iloc[0], q.iloc[1]
    r = hi - lo
    if r == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / r - 0.5) * 2


def compute_rsi(df: pd.DataFrame, period: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """RSI as normalized signal: oversold -> positive, overbought -> negative."""
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Normalize: 30 -> +1 (oversold, expect up), 70 -> -1 (overbought, expect down)
    signal = (50 - rsi) / 50
    return signal.clip(-1, 1)


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """MACD histogram as normalized signal."""
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    hist = macd_line - signal_line
    return _normalize_signal(hist)


def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2) -> pd.Series:
    """Bollinger: below lower band -> positive, above upper -> negative."""
    close = df["close"]
    sma = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # Position within bands: -1 at upper, +1 at lower
    width = upper - lower
    width = width.replace(0, np.nan)
    pos = (close - lower) / width
    signal = (0.5 - pos) * 2
    return signal.clip(-1, 1)


def compute_dual_sma(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """Dual SMA: fast > slow -> positive, fast < slow -> negative."""
    close = df["close"]
    sma_fast = close.rolling(window=fast, min_periods=fast).mean()
    sma_slow = close.rolling(window=slow, min_periods=slow).mean()
    diff = (sma_fast - sma_slow) / sma_slow
    return _normalize_signal(diff)


def compute_mean_reversion(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Mean reversion: below SMA -> positive, above -> negative."""
    close = df["close"]
    sma = close.rolling(window=period, min_periods=period).mean()
    dev = (sma - close) / sma.replace(0, np.nan)
    return _normalize_signal(dev)


def compute_breakout(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Breakout: above channel -> positive, below -> negative."""
    high = df["high"]
    low = df["low"]
    upper = high.rolling(window=period, min_periods=period).max()
    lower = low.rolling(window=period, min_periods=period).min()
    close = df["close"]
    mid = (upper + lower) / 2
    width = (upper - lower).replace(0, np.nan)
    pos = (close - mid) / width
    return _normalize_signal(pos)


def compute_buy_hold(df: pd.DataFrame) -> pd.Series:
    """Buy & hold: always +0.5 (slight bullish)."""
    return pd.Series(0.5, index=df.index)


def compute_vwap(df: pd.DataFrame, band_pct: float = 0.5) -> pd.Series:
    """VWAP deviation signal — optimised for intraday timeframes (1m – 1H).

    VWAP is computed as the session cumulative (typical_price * volume) /
    cumulative volume.  The signal measures how far price has stretched from
    VWAP and fades back toward it:

    * Price well **below** VWAP → signal approaches **+1** (mean-revert long)
    * Price well **above** VWAP → signal approaches **-1** (mean-revert short)

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a datetime index.
    band_pct : float
        Half-band width as a percentage of VWAP (default 0.5 %).  A deviation
        of ``band_pct`` maps to ±1.  Tighten for scalping (0.2), widen for
        swing (1.0).

    Returns
    -------
    pd.Series
        Normalised signal in [-1, 1].

    Notes
    -----
    On daily data VWAP resets once per day, collapsing to the typical price of
    each bar — the indicator still works but is less meaningful.  Prefer
    intraday timeframes (1m, 5m, 15m, 1H) for best results.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan)

    # Group by calendar date so VWAP resets each session.
    dates = df.index.normalize() if hasattr(df.index, "normalize") else pd.to_datetime(df.index).normalize()
    cum_tp_vol = (tp * vol).groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum().clip(lower=1e-9)
    vwap = cum_tp_vol / cum_vol

    dev_pct = (df["close"] - vwap) / vwap.replace(0, np.nan) * 100
    # Clamp: band_pct above → -1, band_pct below → +1
    signal = -(dev_pct / band_pct).clip(-1, 1)
    return signal.fillna(0.0)


def compute_orderflow_vwap(df: pd.DataFrame, atr_period: int = 14, clip_value: float = 2.0) -> pd.Series:
    """Order flow VWAP deviation normalized by ATR."""
    return compute_vwap_signal(df, atr_period=atr_period, clip_value=clip_value)


def compute_volume_profile(df: pd.DataFrame, window: int = 20, bins: int = 16, near_poc_threshold: float = 0.002) -> pd.Series:
    """Rolling volume profile signal using point of control proximity."""
    _poc, signal = compute_volume_profile_signal(df, window=window, bins=bins, near_poc_threshold=near_poc_threshold)
    return signal


def compute_cumulative_delta(df: pd.DataFrame, window: int = 20, clip_value: float = 1.0) -> pd.Series:
    """Rolling cumulative delta signal from the configured order flow provider."""
    provider = get_order_flow_provider()
    flow = provider.get_order_flow(df)
    return compute_cumulative_delta_signal(flow, df["volume"], window=window, clip_value=clip_value)


def compute_liquidity_metrics(df: pd.DataFrame, window: int = 20, amihud_scale: float = 1e6) -> pd.Series:
    """Liquidity signal based on spread proxy and Amihud illiquidity."""
    provider = get_order_flow_provider()
    flow = provider.get_order_flow(df)
    return compute_liquidity_signal(df, flow, amihud_scale=amihud_scale, window=window)

def compute_return_entropy(df: pd.DataFrame, window: int = 20, bins: int = 20, base: float = 2.0) -> pd.Series:
    """Rolling Shannon entropy of returns normalized to [-1, 1]."""
    return compute_entropy_signal(df, window=window, bins=bins, base=base)


def compute_mutual_information(df: pd.DataFrame, window: int = 20, bins: int = 20, mode: str = "price_volume") -> pd.Series:
    """Rolling mutual information signal between returns and volume changes (or returns)."""
    return compute_mutual_info_signal(df, window=window, bins=bins, mode=mode)


def compute_fisher_information(df: pd.DataFrame, window: int = 20, clip_percentile: float = 95.0) -> pd.Series:
    """Rolling Fisher information proxy from inverse return variance."""
    return compute_fisher_information_signal(df, window=window, clip_percentile=clip_percentile)


def compute_kld_regime_shift(
    df: pd.DataFrame,
    recent_window: int = 20,
    reference_window: int = 60,
    bins: int = 20,
    sigmoid_scale: float = 3.0,
) -> pd.Series:
    """Symmetric KL-divergence signal comparing recent vs prior return distributions."""
    return compute_kld_signal(
        df,
        recent_window=recent_window,
        reference_window=reference_window,
        bins=bins,
        sigmoid_scale=sigmoid_scale,
    )


def compute_rolling_entropy(df: pd.DataFrame, window: int = 20, bins: int = 10) -> pd.Series:
    """Shannon entropy of rolling return distributions."""
    returns = df["close"].pct_change().fillna(0.0)

    def _entropy(x: np.ndarray) -> float:
        hist, _ = np.histogram(x, bins=max(2, int(bins)), density=True)
        probs = hist / (hist.sum() + 1e-12)
        probs = probs[probs > 0]
        return float(-(probs * np.log(probs)).sum())

    ent = returns.rolling(int(window), min_periods=max(5, int(window) // 2)).apply(_entropy, raw=True)
    return _normalize_signal(ent.fillna(0.0))


def compute_mutual_information(df: pd.DataFrame, window: int = 30, bins: int = 8, lag: int = 1) -> pd.Series:
    """Rolling mutual information between returns and lagged returns."""
    returns = df["close"].pct_change().fillna(0.0)
    shifted = returns.shift(int(lag)).fillna(0.0)

    def _mi(x: np.ndarray, y: np.ndarray) -> float:
        joint_hist, _, _ = np.histogram2d(x, y, bins=max(2, int(bins)))
        pxy = joint_hist / (joint_hist.sum() + 1e-12)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        expected = px @ py
        mask = pxy > 0
        return float((pxy[mask] * np.log((pxy[mask] + 1e-12) / (expected[mask] + 1e-12))).sum())

    vals = np.full(len(returns), np.nan)
    w = int(window)
    for i in range(w - 1, len(returns)):
        x = returns.iloc[i - w + 1 : i + 1].to_numpy(dtype=float)
        y = shifted.iloc[i - w + 1 : i + 1].to_numpy(dtype=float)
        vals[i] = _mi(x, y)
    mi = pd.Series(vals, index=df.index)
    return _normalize_signal(mi.fillna(0.0))


def compute_fisher_information(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Fisher-like information proxy using squared standardized return slopes."""
    returns = df["close"].pct_change().fillna(0.0)
    z = (returns - returns.rolling(window, min_periods=max(5, window // 2)).mean())
    z = z / returns.rolling(window, min_periods=max(5, window // 2)).std().replace(0.0, np.nan)
    fisher = z.diff().pow(2).rolling(window, min_periods=max(5, window // 2)).mean()
    return _normalize_signal(fisher.fillna(0.0))


def compute_kl_divergence(df: pd.DataFrame, window: int = 30, bins: int = 10) -> pd.Series:
    """KL divergence between consecutive rolling return distributions."""
    returns = df["close"].pct_change().fillna(0.0)
    values = np.full(len(returns), np.nan)
    w = int(window)
    b = max(2, int(bins))
    for i in range(2 * w - 1, len(returns)):
        prev = returns.iloc[i - 2 * w + 1 : i - w + 1].to_numpy(dtype=float)
        curr = returns.iloc[i - w + 1 : i + 1].to_numpy(dtype=float)
        low = float(min(prev.min(), curr.min()))
        high = float(max(prev.max(), curr.max()))
        if low == high:
            values[i] = 0.0
            continue
        p_hist, _ = np.histogram(prev, bins=b, range=(low, high), density=True)
        q_hist, _ = np.histogram(curr, bins=b, range=(low, high), density=True)
        p_dist = p_hist / (p_hist.sum() + 1e-12)
        q_dist = q_hist / (q_hist.sum() + 1e-12)
        values[i] = float(np.sum(p_dist * np.log((p_dist + 1e-12) / (q_dist + 1e-12))))
    kl = pd.Series(values, index=df.index)
    return _normalize_signal(kl.fillna(0.0))


def compute_mft_signal(
    df: pd.DataFrame,
    kernel: str = "gaussian",
    sigma: float = 10.0,
    threshold: float = 0.0,
    smooth_window: int = 1,
) -> pd.Series:
    """Simplified MFT directional signal from field-potential gradient."""
    return mft_signal(
        close=df["close"],
        kernel=kernel,
        sigma=sigma,
        threshold=threshold,
        smooth_window=smooth_window,
    )


def compute_mft_energy(
    df: pd.DataFrame,
    kernel: str = "gaussian",
    sigma: float = 10.0,
    energy_window: int = 20,
) -> pd.Series:
    """MFT energy-derived signal based on relative field activity."""
    return mft_energy_signal(
        close=df["close"],
        kernel=kernel,
        sigma=sigma,
        energy_window=energy_window,
    )




def compute_mft_complex_amplitude(df: pd.DataFrame) -> pd.Series:
    """Instantaneous amplitude from Hilbert analytic signal of close."""
    out = complex_potential(df["close"].astype(float))["amplitude"]
    return _normalize_signal(out.fillna(0.0))


def compute_mft_complex_phase(df: pd.DataFrame) -> pd.Series:
    """Instantaneous phase from Hilbert analytic signal of close."""
    out = complex_potential(df["close"].astype(float))["phase"]
    return _normalize_signal(out.fillna(0.0))


def compute_mft_phase_change(df: pd.DataFrame) -> pd.Series:
    """Phase-difference proxy for instantaneous frequency shifts."""
    out = complex_potential(df["close"].astype(float))["phase_change"]
    return _normalize_signal(out.fillna(0.0))


def compute_mft_price_volume_interaction(
    df: pd.DataFrame,
    kernel: str = "gaussian",
    sigma: float = 10.0,
    corr_window: int = 20,
) -> pd.Series:
    """Price/volume field interaction using potential and gradient coupling."""
    return _normalize_signal(
        volume_price_interaction(
            price_series=df["close"].astype(float),
            volume_series=df["volume"].astype(float),
            kernel=kernel,
            sigma=sigma,
            corr_window=corr_window,
        ).fillna(0.0)
    )


def compute_mft_spectral_power(
    df: pd.DataFrame,
    window: int = 64,
    band: str = "low",
) -> pd.Series:
    """Rolling relative FFT power for a selected frequency band."""
    band_map = {
        "low": (0.0, 0.2),
        "mid": (0.2, 0.5),
        "high": (0.5, 1.0),
    }
    bounds = band_map.get(str(band).lower(), band_map["low"])
    power = rolling_spectral_power(df["close"].astype(float), window=int(window), bands=[bounds])
    signed = 2.0 * power.iloc[:, 0] - 1.0
    return signed.fillna(0.0).clip(-1.0, 1.0)

def _extract_information_flow_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Build a symbol->close matrix from single or multi-symbol inputs."""
    if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return pd.DataFrame({"SYMBOL": df["close"].astype(float)}, index=df.index)
    if isinstance(df.columns, pd.MultiIndex):
        if "close" in df.columns.get_level_values(1):
            return df.xs("close", axis=1, level=1).astype(float)
        if "close" in df.columns.get_level_values(0):
            return df["close"].astype(float)
    return df.astype(float)


def compute_transfer_entropy(
    df: pd.DataFrame,
    window: int = 50,
    from_symbol: str = "SYMBOL",
    to_symbol: str = "SYMBOL",
    bins: int = 3,
    normalize: bool = True,
) -> pd.Series:
    """Compute rolling transfer entropy for selected pair."""
    prices = _extract_information_flow_prices(df)
    return rolling_transfer_entropy(
        prices=prices,
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        window=window,
        bins=bins,
        normalize=normalize,
    ).fillna(0.0)


def compute_granger_causality(
    df: pd.DataFrame,
    window: int = 50,
    from_symbol: str = "SYMBOL",
    to_symbol: str = "SYMBOL",
    maxlags: int = 2,
    threshold: float = 0.05,
    output: str = "pvalue",
) -> pd.Series:
    """Compute rolling Granger-causality output for selected pair."""
    prices = _extract_information_flow_prices(df)
    return rolling_granger_causality(
        prices=prices,
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        window=window,
        maxlags=maxlags,
        threshold=threshold,
        output=output,
    ).fillna(0.0)


INDICATOR_COMPUTERS: dict[str, Callable[..., pd.Series]] = {
    "RSI": compute_rsi,
    "MACD": compute_macd,
    "Bollinger": compute_bollinger,
    "Dual SMA": compute_dual_sma,
    "Mean Reversion": compute_mean_reversion,
    "Breakout": compute_breakout,
    "Buy & Hold": compute_buy_hold,
    "VWAP": compute_vwap,
    "Orderflow VWAP": compute_orderflow_vwap,
    "Volume Profile": compute_volume_profile,
    "Cumulative Delta": compute_cumulative_delta,
    "Liquidity Metrics": compute_liquidity_metrics,
    "Rolling Entropy": compute_rolling_entropy,
    "Mutual Information": compute_mutual_information,
    "Fisher Information": compute_fisher_information,
    "KL Divergence": compute_kl_divergence,
    "MFT Signal": compute_mft_signal,
    "MFT Energy": compute_mft_energy,
    "MFT Complex Amplitude": compute_mft_complex_amplitude,
    "MFT Complex Phase": compute_mft_complex_phase,
    "MFT Phase Change": compute_mft_phase_change,
    "MFT Price-Volume Interaction": compute_mft_price_volume_interaction,
    "MFT Spectral Power": compute_mft_spectral_power,
    "Phi-Bot (MFT)": compute_mft_signal,
    "Return Entropy": compute_return_entropy,
    "Mutual Information": compute_mutual_information,
    "Fisher Information": compute_fisher_information,
    "KL Divergence": compute_kld_regime_shift,
    "Transfer Entropy": compute_transfer_entropy,
    "Granger Causality": compute_granger_causality,
}


_PARAM_MAP = {
    "RSI": {"rsi_period": "period", "oversold": "oversold", "overbought": "overbought"},
    "MACD": {"fast_period": "fast", "slow_period": "slow", "signal_period": "signal_period"},
    "Bollinger": {"bb_period": "period", "num_std": "num_std"},
    "Dual SMA": {"fast_period": "fast", "slow_period": "slow"},
    "Mean Reversion": {"sma_period": "period"},
    "Breakout": {"channel_period": "period"},
    "VWAP": {"band_pct": "band_pct"},
    "Orderflow VWAP": {"atr_period": "atr_period", "clip_value": "clip_value"},
    "Volume Profile": {"window": "window", "bins": "bins", "near_poc_threshold": "near_poc_threshold"},
    "Cumulative Delta": {"window": "window", "clip_value": "clip_value"},
    "Liquidity Metrics": {"window": "window", "amihud_scale": "amihud_scale"},
    "Rolling Entropy": {"window": "window", "bins": "bins"},
    "Mutual Information": {"window": "window", "bins": "bins", "lag": "lag"},
    "Fisher Information": {"window": "window"},
    "KL Divergence": {"window": "window", "bins": "bins"},
    "MFT Signal": {"kernel": "kernel", "sigma": "sigma", "threshold": "threshold", "smooth_window": "smooth_window"},
    "MFT Energy": {"kernel": "kernel", "sigma": "sigma", "energy_window": "energy_window"},
    "MFT Complex Amplitude": {},
    "MFT Complex Phase": {},
    "MFT Phase Change": {},
    "MFT Price-Volume Interaction": {"kernel": "kernel", "sigma": "sigma", "corr_window": "corr_window"},
    "MFT Spectral Power": {"window": "window", "band": "band"},
    "Phi-Bot (MFT)": {},
    "Return Entropy": {"window": "window", "bins": "bins", "base": "base"},
    "Mutual Information": {"window": "window", "bins": "bins", "mode": "mode"},
    "Fisher Information": {"window": "window", "clip_percentile": "clip_percentile"},
    "KL Divergence": {"recent_window": "recent_window", "reference_window": "reference_window", "bins": "bins", "sigmoid_scale": "sigmoid_scale"},
    "Transfer Entropy": {"window": "window", "from_symbol": "from_symbol", "to_symbol": "to_symbol", "bins": "bins", "normalize": "normalize"},
    "Granger Causality": {"window": "window", "from_symbol": "from_symbol", "to_symbol": "to_symbol", "maxlags": "maxlags", "threshold": "threshold", "output": "output"},
}


def compute_indicator(name: str, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Compute indicator signal by name with params."""
    fn = INDICATOR_COMPUTERS.get(name)
    if fn is None:
        return pd.Series(0.0, index=df.index)
    pmap = _PARAM_MAP.get(name, {})
    kwargs = {pmap.get(k, k): v for k, v in params.items()}
    try:
        return fn(df, **kwargs)
    except Exception:
        return pd.Series(0.0, index=df.index)

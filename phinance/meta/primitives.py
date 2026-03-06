"""Primitive set definition for GP-based strategy discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from deap import gp


@dataclass(frozen=True)
class PrimitiveContext:
    """Context describing GP input feature ordering."""

    feature_names: List[str]


def protected_div(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    """Vectorized protected division returning zero where denominator is near zero."""
    numerator = np.asarray(left, dtype=float)
    denominator = np.asarray(right, dtype=float)
    safe = np.where(np.abs(denominator) < 1e-8, 1.0, denominator)
    out = numerator / safe
    return np.where(np.isfinite(out), out, 0.0)


def safe_log(value: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.log(np.clip(np.abs(arr), 1e-8, None))


def safe_sqrt(value: np.ndarray | float) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(value, dtype=float), 0.0, None))


def to_float_mask(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value, dtype=float) > 0).astype(float)


def if_then_else(cond: np.ndarray | float, a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.where(np.asarray(cond, dtype=float) > 0, np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def random_constant() -> float:
    return float(np.random.uniform(-1.0, 1.0))


def build_feature_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build numeric features available as GP terminals."""
    close = ohlcv["close"].astype(float)
    volume = ohlcv.get("volume", pd.Series(1.0, index=ohlcv.index)).astype(float)
    ret_1 = close.pct_change().fillna(0.0)
    ret_5 = close.pct_change(5).fillna(0.0)
    momentum_10 = close / close.rolling(10, min_periods=1).mean() - 1.0
    vol_10 = ret_1.rolling(10, min_periods=1).std().fillna(0.0)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0.0).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi_14 = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0) / 100.0

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = (ema_fast - ema_slow).fillna(0.0)

    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std().fillna(0.0)
    bb_upper_dist = ((bb_mid + 2.0 * bb_std) / close - 1.0).fillna(0.0)
    bb_lower_dist = ((bb_mid - 2.0 * bb_std) / close - 1.0).fillna(0.0)

    feature_df = pd.DataFrame(
        {
            "close": close,
            "volume": volume,
            "returns_1": ret_1,
            "returns_5": ret_5,
            "momentum_10": momentum_10.fillna(0.0),
            "volatility_10": vol_10,
            "rsi_14": rsi_14,
            "macd": macd,
            "bb_upper_dist": bb_upper_dist,
            "bb_lower_dist": bb_lower_dist,
        },
        index=ohlcv.index,
    )
    return feature_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_primitive_set(feature_names: Iterable[str]) -> tuple[gp.PrimitiveSet, PrimitiveContext]:
    """Create a DEAP primitive set with arithmetic and logical operators."""
    names = list(feature_names)
    pset = gp.PrimitiveSet("MAIN", len(names))
    for idx, name in enumerate(names):
        pset.renameArguments(**{f"ARG{idx}": name})

    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.tanh, 1)
    pset.addPrimitive(safe_log, 1)
    pset.addPrimitive(safe_sqrt, 1)

    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(if_then_else, 3)
    pset.addPrimitive(to_float_mask, 1)

    pset.addEphemeralConstant("rand", random_constant)
    pset.addTerminal(0.0)
    pset.addTerminal(1.0)
    pset.addTerminal(-1.0)

    return pset, PrimitiveContext(feature_names=names)

"""Synthetic order-flow generators for LOB research backtests."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from phi.lob.data import LobEvent


def generate_synthetic_events(
    *,
    start_price: float = 100.0,
    spread_bps: float = 2.0,
    arrival_rate: float = 20.0,
    volatility: float = 0.01,
    size_mean: float = 0.0,
    size_sigma: float = 0.5,
    seed: int | None = None,
) -> Iterator[LobEvent]:
    """Yield synthetic LOB events using Poisson timing and GBM-like price dynamics."""
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now(tz="UTC")
    mid = float(start_price)

    while True:
        dt = float(rng.exponential(1.0 / max(arrival_rate, 1e-6)))
        now += pd.to_timedelta(dt, unit="s")
        drift = -0.5 * volatility * volatility * dt
        shock = volatility * np.sqrt(dt) * float(rng.standard_normal())
        mid = max(1e-6, mid * float(np.exp(drift + shock)))

        side = "buy" if rng.random() < 0.5 else "sell"
        event_kind = rng.choice(["add", "cancel", "trade"], p=[0.65, 0.2, 0.15])
        spread = mid * spread_bps / 10_000.0
        price = mid - spread / 2.0 if side == "buy" else mid + spread / 2.0
        volume = float(max(1.0, rng.lognormal(mean=size_mean, sigma=max(size_sigma, 1e-6))))

        yield LobEvent(
            timestamp=now,
            event_type=str(event_kind),
            price=float(price),
            volume=volume,
            side=side,
            order_id=None,
        )

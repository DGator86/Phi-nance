"""
Lumibot Strategy that uses Phi-nance projection pipeline (assign -> engines -> MFM -> composer).

Signal: daily direction from ProjectionPacket (UP/DOWN). Optional: submit orders or only record.
Requires lumibot: pip install phi-nance[lumibot].
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def create_projection_strategy_class(
    bar_store: Any,
    assigner: Any,
    engines: dict[str, Any],
    composer: Any,
    tickers: list[str],
) -> type:
    """
    Build a Lumibot Strategy class that uses the given bar_store, assigner, engines, composer.
    The strategy runs once per day (sleeptime 1D), gets ProjectionPacket per ticker, and goes long
    on UP and sells on DOWN (or records only, no orders).
    """
    try:
        from lumibot.strategies import Strategy
    except ImportError as e:
        raise ImportError("Lumibot is required. Install with: pip install phi-nance[lumibot]") from e

    from phinence.contracts.projection_packet import Horizon
    from phinence.mfm.merger import build_mfm

    _bar_store = bar_store
    _assigner = assigner
    _engines = engines
    _composer = composer
    _tickers = list(tickers)

    class ProjectionStrategy(Strategy):
        def initialize(self, parameters: dict | None = None):
            self.sleeptime = "1D"
            self._last_signal: dict[str, str] = {}

        def on_trading_iteration(self):
            dt = self.get_datetime()
            if not dt:
                return
            try:
                as_of = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
            except Exception:
                as_of = dt
            end_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
            start_ts = pd.Timestamp(as_of) - pd.Timedelta(days=180)
            for ticker in _tickers:
                try:
                    packet = _assigner.assign(ticker, as_of, start_ts=start_ts, end_ts=end_ts)
                    liq = _engines.get("liquidity")
                    reg = _engines.get("regime")
                    sent = _engines.get("sentiment")
                    hed = _engines.get("hedge")
                    if liq:
                        liq = liq.run(packet)
                    if reg:
                        reg = reg.run(packet)
                    if sent:
                        sent = sent.run(packet)
                    if hed:
                        hed = hed.run(packet)
                    mfm = build_mfm(ticker, as_of, liquidity=liq, regime=reg, sentiment=sent, hedge=hed)
                    proj = _composer.run(mfm, horizons=[Horizon.DAILY])
                    hp = proj.get_horizon(Horizon.DAILY)
                    if not hp:
                        continue
                    direction = "UP" if hp.direction.up >= hp.direction.down else "DOWN"
                    self._last_signal[ticker] = direction
                    # Optional: trade. Buy on UP, sell on DOWN.
                    pos = self.get_position(ticker)
                    if direction == "UP" and (pos is None or pos.quantity <= 0):
                        order = self.create_order(ticker, 100, "buy")
                        if order:
                            self.submit_order(order)
                    elif direction == "DOWN" and pos is not None and pos.quantity > 0:
                        self.sell_all(ticker)
                except Exception:
                    continue

    return ProjectionStrategy

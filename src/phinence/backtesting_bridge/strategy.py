"""
backtesting.py Strategy that uses Phi-nance projection pipeline for signals.

Each bar (e.g. daily): run assign -> engines -> MFM -> composer; if direction UP go long, if DOWN go short/flat.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def create_projection_strategy(
    bar_store: Any,
    assigner: Any,
    engines: dict[str, Any],
    composer: Any,
    ticker: str,
):
    """
    Returns a backtesting.Strategy subclass that uses the projection pipeline each bar.

    Use with daily data (timeframe='1D'). In next(), runs pipeline for current bar's date,
    then buys on UP and sells on DOWN.
    """
    try:
        from backtesting import Strategy
    except ImportError as e:
        raise ImportError("backtesting.py is required. Install with: pip install phi-nance[backtesting]") from e

    from phinence.contracts.projection_packet import Horizon
    from phinence.mfm.merger import build_mfm

    _bar_store = bar_store
    _assigner = assigner
    _engines = engines
    _composer = composer
    _ticker = ticker

    class ProjectionStrategy(Strategy):
        def init(self):
            pass

        def next(self):
            if len(self.data) < 2:
                return
            # Current bar's date (last closed bar in backtesting.py is self.data.index[-1] in next())
            idx = self.data.df.index
            if idx is None or len(idx) == 0:
                return
            # In next(), we're at the bar that just closed; backtesting.py passes data up to and including current
            current = idx[-1]
            try:
                as_of = current.to_pydatetime() if hasattr(current, "to_pydatetime") else current
            except Exception:
                as_of = current
            end_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
            start_ts = pd.Timestamp(as_of) - pd.Timedelta(days=180)
            try:
                packet = _assigner.assign(_ticker, as_of, start_ts=start_ts, end_ts=end_ts)
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
                mfm = build_mfm(_ticker, as_of, liquidity=liq, regime=reg, sentiment=sent, hedge=hed)
                proj = _composer.run(mfm, horizons=[Horizon.DAILY])
                hp = proj.get_horizon(Horizon.DAILY)
                if not hp:
                    return
                direction = "UP" if hp.direction.up >= hp.direction.down else "DOWN"
                if direction == "UP" and not self.position:
                    self.buy()
                elif direction == "DOWN" and self.position:
                    self.position.close()
            except Exception:
                pass

    return ProjectionStrategy

"""Options backtesting engine for multi-leg strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from phi.backtest.engine import BacktestEngine
from phi.options.iv_surface import IVSurface
from phi.options.models.black_scholes import greeks, price_european


@dataclass
class OptionPosition:
    strategy: Any
    quantity: int
    entry_index: int = 0


class OptionsBacktestEngine(BacktestEngine):
    def run(self, config, data: pd.DataFrame) -> Dict[str, Any]:
        if data is None or data.empty:
            raise ValueError("data must contain underlying price history")
        closes = data["close"].astype(float).to_numpy()
        r = float(getattr(config, "extra", {}).get("risk_free_rate", 0.01))
        default_iv = float(getattr(config, "extra", {}).get("default_iv", 0.2))
        chain_data = getattr(config, "extra", {}).get("iv_chain_data")
        iv_surface = IVSurface(chain_data) if chain_data is not None else None

        strategies = getattr(config, "extra", {}).get("options_strategies", [])
        positions: List[OptionPosition] = [
            OptionPosition(strategy=s["strategy"], quantity=int(s.get("quantity", 1)), entry_index=int(s.get("entry_index", 0)))
            for s in strategies
        ]
        if not positions:
            return {"portfolio_value": [config.initial_capital], "total_return": 0.0, "trade_log": [], "greeks": []}

        portfolio_values = []
        greek_series = []
        trade_log = []
        initial_cap = float(config.initial_capital)

        for idx, spot in enumerate(closes):
            port_val = initial_cap
            agg = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
            for pos in positions:
                if idx < pos.entry_index:
                    continue
                for leg in pos.strategy.legs():
                    T = max(float(leg.expiry) - idx / 252.0, 0.0)
                    sigma = iv_surface.get_iv(leg.strike, max(T, 1 / 365)) if iv_surface else default_iv
                    px = price_european(leg.option_type, float(spot), float(leg.strike), T, r, sigma)
                    gs = greeks(leg.option_type, float(spot), float(leg.strike), max(T, 1 / 365), r, sigma)
                    sign = 1.0 if leg.action == "buy" else -1.0
                    scale = sign * leg.quantity * pos.quantity * 100.0
                    port_val += scale * px
                    for k in agg:
                        agg[k] += scale * gs[k]
            portfolio_values.append(port_val)
            greek_series.append(agg)

        total_return = (portfolio_values[-1] - initial_cap) / initial_cap if initial_cap else 0.0
        trade_log.append({"event": "complete", "positions": len(positions), "final_value": portfolio_values[-1]})
        returns = np.diff(np.array(portfolio_values)) / np.maximum(np.array(portfolio_values[:-1]), 1e-12)
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0.0

        return {
            "portfolio_value": portfolio_values,
            "total_return": float(total_return),
            "sharpe": sharpe,
            "trade_log": trade_log,
            "greeks": greek_series,
        }

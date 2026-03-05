"""Options backtesting engine for multi-leg strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from phi.backtest.engine import BacktestEngine
from phi.options.early_exercise import should_exercise_early
from phi.options.iv_surface import HistoricalIVSurface, IVSurface
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
        n = len(closes)
        r = float(getattr(config, "extra", {}).get("risk_free_rate", 0.01))
        default_iv = float(getattr(config, "extra", {}).get("default_iv", 0.2))
        chain_data = getattr(config, "extra", {}).get("iv_chain_data")
        if chain_data is not None:
            iv_surface = IVSurface(chain_data)
        else:
            iv_surface = HistoricalIVSurface(
                symbol=str(getattr(config, "symbols", ["SPY"])[0]),
                start_date=data.index.min(),
                end_date=data.index.max(),
                price_history=data,
            )

        strategies = getattr(config, "extra", {}).get("options_strategies", [])
        positions: List[OptionPosition] = [
            OptionPosition(strategy=s["strategy"], quantity=int(s.get("quantity", 1)), entry_index=int(s.get("entry_index", 0)))
            for s in strategies
        ]
        if not positions:
            return {"portfolio_value": [config.initial_capital], "total_return": 0.0, "trade_log": [], "greeks": []}

        initial_cap = float(config.initial_capital)
        portfolio_values = np.full(n, initial_cap, dtype=float)
        greek_series = np.zeros((n, 5), dtype=float)  # d,g,t,v,r
        trade_log = []

        for pos_idx, pos in enumerate(positions):
            for leg_idx, leg in enumerate(pos.strategy.legs()):
                idxs = np.arange(pos.entry_index, n)
                if idxs.size == 0:
                    continue
                T = np.maximum(float(leg.expiry) - (idxs - pos.entry_index) / 252.0, 0.0)
                scale = (1.0 if leg.action == "buy" else -1.0) * leg.quantity * pos.quantity * 100.0
                for i_rel, i in enumerate(idxs):
                    t_val = float(max(T[i_rel], 1 / 365))
                    if hasattr(iv_surface, "get_iv"):
                        try:
                            sigma = float(iv_surface.get_iv(data.index[i], float(leg.strike), t_val))
                        except TypeError:
                            sigma = float(iv_surface.get_iv(float(leg.strike), t_val))
                    else:
                        sigma = default_iv

                    px = price_european(leg.option_type, float(closes[i]), float(leg.strike), float(T[i_rel]), r, sigma)
                    gs = greeks(leg.option_type, float(closes[i]), float(leg.strike), t_val, r, sigma)

                    is_american = bool(getattr(leg, "american", False) or getattr(pos.strategy, "american", False))
                    if is_american and should_exercise_early(leg.option_type, float(closes[i]), float(leg.strike), float(T[i_rel])):
                        intrinsic = max(float(closes[i]) - float(leg.strike), 0.0) if leg.option_type == "call" else max(float(leg.strike) - float(closes[i]), 0.0)
                        px = max(px, intrinsic)

                    portfolio_values[i] += scale * px
                    greek_series[i, 0] += scale * gs["delta"]
                    greek_series[i, 1] += scale * gs["gamma"]
                    greek_series[i, 2] += scale * gs["theta"]
                    greek_series[i, 3] += scale * gs["vega"]
                    greek_series[i, 4] += scale * gs["rho"]

                trade_log.append({"event": "leg_valued", "position": pos_idx, "leg": leg_idx, "points": int(idxs.size)})

        total_return = (portfolio_values[-1] - initial_cap) / initial_cap if initial_cap else 0.0
        returns = np.diff(portfolio_values) / np.maximum(portfolio_values[:-1], 1e-12)
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0.0

        return {
            "portfolio_value": portfolio_values.tolist(),
            "total_return": float(total_return),
            "sharpe": sharpe,
            "trade_log": trade_log,
            "greeks": [
                {"delta": float(d), "gamma": float(g), "theta": float(t), "vega": float(v), "rho": float(rh)}
                for d, g, t, v, rh in greek_series
            ],
        }

"""
Phi-nance Options Module
========================

Options backtest mode:
  - Simple delta-based simulation (backtest.py)
  - Full walk-forward engine backtest (engine_backtest.py)
  - AI-powered strategy advisor (ai_advisor.py)
  - Post-trade reviewer with regime/IV/GEX insights (options_reviewer.py)

Market data:
  - MarketDataApp real chain snapshots (market_data.py)
"""

from .backtest import run_options_backtest, compute_greeks
from .market_data import get_marketdataapp_snapshot
from .engine_backtest import run_engine_backtest
from .ai_advisor import OptionsAIAdvisor, OptionsRecommendation
from .options_reviewer import review_options_backtest, OptionsBacktestReview
from .models import black_scholes_price, black_scholes_greeks, Greeks

__all__ = [
    "run_options_backtest",
    "compute_greeks",
    "get_marketdataapp_snapshot",
    "run_engine_backtest",
    "OptionsAIAdvisor",
    "OptionsRecommendation",
    "review_options_backtest",
    "OptionsBacktestReview",
    "black_scholes_price",
    "black_scholes_greeks",
    "Greeks",
]

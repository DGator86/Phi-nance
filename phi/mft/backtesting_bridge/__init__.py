"""
backtesting.py (kernc) integration — lightweight backtests with Phi-nance bar data.

Use optional dependency: pip install phi-nance[backtesting]

- bar_store_to_bt_df: convert bar store to OHLCV DataFrame (Open, High, Low, Close, Volume) for Backtest()
- create_projection_strategy: returns a backtesting.Strategy class that uses our projection pipeline
"""

from phi.mft.backtesting_bridge.data import bar_store_to_bt_df
from phi.mft.backtesting_bridge.strategy import create_projection_strategy

__all__ = ["bar_store_to_bt_df", "create_projection_strategy"]

"""
Lumibot integration for backtesting using Phi-nance bar data and projection pipeline.

Use optional dependency: pip install phi-nance[lumibot]

- bar_store_to_pandas_data: convert ParquetBarStore/InMemoryBarStore -> Lumibot pandas_data dict
- create_projection_strategy_class: returns a Lumibot Strategy class that uses our pipeline for signals
"""

from phinence.lumibot_bridge.data import bar_store_to_pandas_data
from phinence.lumibot_bridge.strategy import create_projection_strategy_class

__all__ = ["bar_store_to_pandas_data", "create_projection_strategy_class"]

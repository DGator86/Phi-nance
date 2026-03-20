"""
Results database for storing backtest outcomes.
Uses SQLite for simplicity; can be swapped for PostgreSQL later.
"""
import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


class ResultsDB:
    """
    Manages storage and retrieval of backtest results.
    Each run is stored with metadata, parameters, and performance metrics.
    """

    def __init__(self, db_path: str = "backtest_results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,           -- when the backtest was run
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    parameters TEXT,                    -- JSON dict of parameters
                    regime_sequence TEXT,               -- JSON list of regimes encountered daily
                    start_date TEXT,
                    end_date TEXT,
                    initial_cash REAL,
                    final_cash REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    num_trades INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol ON backtest_runs(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strategy ON backtest_runs(strategy_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_regime ON backtest_runs(regime_sequence)
            """)

    def insert_run(self, run_data: Dict[str, Any]):
        """
        Insert a new backtest run.
        Expected keys:
            timestamp, symbol, strategy_name, parameters (dict),
            regime_sequence (list), start_date, end_date, initial_cash,
            final_cash, total_return, sharpe_ratio, max_drawdown,
            win_rate, num_trades
        """
        # Convert dicts/lists to JSON strings
        run_data = run_data.copy()
        if 'parameters' in run_data and isinstance(run_data['parameters'], dict):
            run_data['parameters'] = json.dumps(run_data['parameters'])
        if 'regime_sequence' in run_data and isinstance(run_data['regime_sequence'], list):
            run_data['regime_sequence'] = json.dumps(run_data['regime_sequence'])

        columns = ', '.join(run_data.keys())
        placeholders = ', '.join(['?' for _ in run_data])
        values = list(run_data.values())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                INSERT INTO backtest_runs ({columns})
                VALUES ({placeholders})
            """, values)

    def get_runs(self, symbol: Optional[str] = None,
                 strategy_name: Optional[str] = None,
                 limit: int = 100) -> pd.DataFrame:
        """
        Retrieve runs as a DataFrame, optionally filtered.
        """
        query = "SELECT * FROM backtest_runs"
        conditions = []
        params = []
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if strategy_name:
            conditions.append("strategy_name = ?")
            params.append(strategy_name)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        # Convert JSON columns back to Python objects
        if not df.empty:
            df['parameters'] = df['parameters'].apply(lambda x: json.loads(x) if x else {})
            df['regime_sequence'] = df['regime_sequence'].apply(lambda x: json.loads(x) if x else [])
        return df

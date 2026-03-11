"""PostgreSQL-backed vendor for intraday options data."""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from phi.config import get_settings
from phi.logging import get_logger

logger = get_logger(__name__)


class PostgresOptionsVendor:
    """Fetch options-chain-like rows from a PostgreSQL ticker table."""

    def __init__(self) -> None:
        settings = get_settings()
        self.engine = create_engine(
            (
                f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
        )

    @staticmethod
    def _safe_table_name(symbol: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_]", "", symbol).lower()
        if not cleaned:
            raise ValueError("symbol must resolve to a non-empty table name")
        return cleaned

    def fetch(self, symbol: str, start: str, end: str, **kwargs) -> pd.DataFrame:
        """Fetch rows between start/end from the symbol-named table.

        Expected timestamp column defaults to ``quote_time`` in milliseconds.
        Override via ``timestamp_column`` or ``timestamp_unit`` kwargs.
        """
        table_name = self._safe_table_name(symbol)
        ts_col = kwargs.get("timestamp_column", "quote_time")
        ts_unit = kwargs.get("timestamp_unit", "ms")

        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())
        if ts_unit == "ms":
            start_ts *= 1000
            end_ts *= 1000
        elif ts_unit == "ns":
            start_ts *= 1_000_000_000
            end_ts *= 1_000_000_000

        query = text(
            f"""
            SELECT *
            FROM {table_name}
            WHERE {ts_col} BETWEEN :start_ts AND :end_ts
            ORDER BY {ts_col}
            """
        )

        logger.info("Fetching postgres options data for %s from %s to %s", symbol, start, end)
        try:
            df = pd.read_sql(query, self.engine, params={"start_ts": start_ts, "end_ts": end_ts})
        except SQLAlchemyError as exc:
            raise ValueError(f"PostgreSQL query failed for {symbol}: {exc}") from exc

        if df.empty:
            logger.warning("No postgres rows returned for %s in %s to %s", symbol, start, end)
        return df

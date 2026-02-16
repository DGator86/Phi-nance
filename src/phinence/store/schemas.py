"""Arrow schemas for bar and chain data. Shared by historical and live."""

import pyarrow as pa

# 1m OHLCV bar
BAR_1M_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="America/New_York")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
])

# 5m OHLCV bar (same shape; resampled from 1m)
BAR_5M_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="America/New_York")),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
])

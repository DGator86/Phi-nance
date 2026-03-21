#!/usr/bin/env python3
"""Export Phi-nance OHLCV parquet for TensorTrade / external RL pipelines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--timeframe", default="1D")
    p.add_argument("--vendor", default="yfinance")
    p.add_argument("--out", required=True, help="Output .parquet path")
    p.add_argument(
        "--cached-only",
        action="store_true",
        help="Use load_ohlcv_cached_first (no network if cache miss)",
    )
    args = p.parse_args()

    from phi.data.unified_data import get_ohlcv, load_ohlcv_cached_first

    if args.cached_only:
        df = load_ohlcv_cached_first(
            args.symbol, args.start, args.end, args.timeframe, args.vendor
        )
    else:
        df = get_ohlcv(
            args.symbol, args.start, args.end, args.timeframe, args.vendor
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

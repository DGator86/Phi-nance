#!/usr/bin/env python3
"""Export per-bar regime labels as CSV for QuantConnect custom data (Phi-nance trained detector)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to trained regime .pkl")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--timeframe", default="1D")
    p.add_argument("--vendor", default="yfinance")
    p.add_argument("--out", required=True, help="Output CSV path (time,regime)")
    args = p.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    import pandas as pd

    from phi.data.unified_data import get_ohlcv
    from phi.regime import load_detector

    ohlcv = get_ohlcv(args.symbol, args.start, args.end, args.timeframe, args.vendor)
    det = load_detector(args.model)
    regimes = det.predict(ohlcv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(regimes.index).strftime("%Y-%m-%d"),
            "regime": regimes.values,
        }
    )
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out.resolve()}")


if __name__ == "__main__":
    main()

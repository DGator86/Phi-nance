#!/usr/bin/env python3
"""Export a QuantConnect-oriented bundle (OHLCV CSV + manifest + optional signal card)."""

from __future__ import annotations

import argparse
import json
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
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (created if missing), e.g. ./exports/qc_SPY_1D",
    )
    p.add_argument(
        "--include-signal-card",
        action="store_true",
        help="Run options signal generator and write signal_card.json",
    )
    p.add_argument(
        "--no-enrich-uw",
        action="store_true",
        help="Skip Unusual Whales chain enrich on signal card",
    )
    args = p.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    from phi.data.ohlcv_fallback import fetch_ohlcv_uw_then_yf
    from phi.integrations.quantconnect import write_quantconnect_bundle
    from phi.logging import get_logger, setup_logging

    setup_logging()
    log = get_logger("scripts.export_quantconnect_bundle")

    df, vendor = fetch_ohlcv_uw_then_yf(
        args.symbol, args.start, args.end, timeframe=args.timeframe
    )
    out = Path(args.out_dir)
    signal_card: dict | None = None
    if args.include_signal_card:
        from phi.options.signal_generator import build_options_signal_card

        card = build_options_signal_card(
            df,
            symbol=args.symbol.strip().upper(),
            ohlcv_vendor=vendor,
            enrich_unusual_whales_chain=not args.no_enrich_uw,
        )
        signal_card = card.model_dump(mode="json")

    write_quantconnect_bundle(
        out,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        ohlcv_vendor=vendor,
        df=df,
        signal_card=signal_card,
    )
    log.info(
        "Wrote QC bundle symbol=%s vendor=%s rows=%s dir=%s",
        args.symbol.upper(),
        vendor,
        len(df),
        out,
    )
    print(json.dumps({"out_dir": str(out.resolve()), "rows": len(df), "vendor": vendor}))


if __name__ == "__main__":
    main()

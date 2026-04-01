#!/usr/bin/env python3
"""Build Lean US equity daily spy.zip from Phi-nance ohlcv.csv.

Lean expects (QuantConnect Lean Data/equity/readme.md):
  <data-folder>/equity/usa/daily/spy.zip containing spy.csv with lines:

    YYYYMMDD HH:MM,open,high,low,close,volume

Prices are decicents (dollars * 10000). We use 00:00 on the export calendar date.

Usage:
    python scripts/lean_spy_daily_zip_from_ohlcv.py --ohlcv path/to/ohlcv.csv --out-zip path/to/spy.zip
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from datetime import datetime
from pathlib import Path


def _to_decicent(x: float) -> int:
    return int(round(float(x) * 10_000))


def build_spy_daily_zip(ohlcv_csv: Path, out_zip: Path) -> int:
    """Read Phi-nance ohlcv.csv; write Lean spy.zip. Returns row count."""
    lines_out: list[str] = []
    with ohlcv_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Empty CSV: {ohlcv_csv}")
        h0 = header[0].strip().lower()
        if h0 != "time":
            raise ValueError(f"Expected time column first, got header: {header}")
        for row in reader:
            if not row or not row[0] or not row[0][0].isdigit():
                continue
            day = datetime.strptime(row[0].strip()[:10], "%Y-%m-%d")
            o, h, low, c, vol = map(float, row[1:6])
            dt = day.strftime("%Y%m%d") + " 00:00"
            v = max(0, int(round(vol)))
            lines_out.append(
                f"{dt},{_to_decicent(o)},{_to_decicent(h)},{_to_decicent(low)},"
                f"{_to_decicent(c)},{v}"
            )

    if not lines_out:
        raise ValueError(f"No data rows in {ohlcv_csv}")

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    inner = "spy.csv"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner, "\n".join(lines_out) + "\n")

    return len(lines_out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build Lean spy.zip from ohlcv.csv")
    p.add_argument("--ohlcv", type=Path, required=True)
    p.add_argument("--out-zip", type=Path, required=True)
    args = p.parse_args()
    n = build_spy_daily_zip(args.ohlcv.resolve(), args.out_zip.resolve())
    print(f"Wrote {n} rows -> {args.out_zip}")


if __name__ == "__main__":
    main()

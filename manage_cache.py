#!/usr/bin/env python3
"""
manage_cache.py — Phi-nance cache management CLI.

Usage
-----
List all cached datasets::

    python manage_cache.py --list

Remove parquet files older than N days::

    python manage_cache.py --clean --days 30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA_CACHE_ROOT = _ROOT / "data_cache"


def cmd_list() -> None:
    """Print all cached datasets with metadata."""
    from phi.data.cache import list_cached_datasets

    datasets = list_cached_datasets()
    if not datasets:
        print("No cached datasets found.")
        return
    print(f"{'Vendor':<15} {'Symbol':<10} {'TF':<6} {'Start':<12} {'End':<12} {'Rows':<8} {'Fetched At'}")
    print("-" * 90)
    for d in datasets:
        print(
            f"{d.get('vendor', ''):<15} "
            f"{d.get('symbol', ''):<10} "
            f"{d.get('timeframe', ''):<6} "
            f"{d.get('start', ''):<12} "
            f"{d.get('end', ''):<12} "
            f"{d.get('rows', ''):<8} "
            f"{d.get('fetched_at', 'unknown')}"
        )


def cmd_clean(days: int) -> None:
    """Remove parquet files older than *days* days."""
    if not _DATA_CACHE_ROOT.exists():
        print("Cache directory does not exist.")
        return

    now = datetime.now(tz=timezone.utc)
    removed = 0
    for parquet in _DATA_CACHE_ROOT.rglob("*.parquet"):
        age_days = (now - datetime.fromtimestamp(parquet.stat().st_mtime, tz=timezone.utc)).total_seconds() / 86400
        if age_days > days:
            meta = Path(str(parquet) + ".metadata.json")
            parquet.unlink()
            if meta.exists():
                meta.unlink()
            print(f"Removed: {parquet.relative_to(_DATA_CACHE_ROOT)} ({age_days:.1f} days old)")
            removed += 1

    print(f"\nRemoved {removed} file(s) older than {days} days.")


def main() -> None:
    """Entry point for manage_cache CLI."""
    parser = argparse.ArgumentParser(
        description="Phi-nance cache management utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true", help="List all cached datasets.")
    parser.add_argument("--clean", action="store_true", help="Remove cached files older than --days.")
    parser.add_argument("--days", type=int, default=30, help="Age threshold in days for --clean (default: 30).")
    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.clean:
        cmd_clean(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

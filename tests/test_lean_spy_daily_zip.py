"""Lean spy.zip builder from Phi-nance ohlcv.csv (stdlib only)."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_lean_spy_daily_zip_from_ohlcv_cli(tmp_path: Path) -> None:
    csv = tmp_path / "ohlcv.csv"
    csv.write_text(
        "time,open,high,low,close,volume\n"
        "2024-01-02,10.0,11.0,9.0,10.5,1000.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "spy.zip"
    script = Path(__file__).resolve().parents[1] / "scripts" / "lean_spy_daily_zip_from_ohlcv.py"
    subprocess.run(
        [sys.executable, str(script), "--ohlcv", str(csv), "--out-zip", str(out)],
        check=True,
    )
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "spy.csv" in names
        body = zf.read("spy.csv").decode("utf-8")
    assert "20240102 00:00" in body
    assert ",105000," in body  # 10.5 * 10000 decicents

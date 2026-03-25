"""Export OHLCV + metadata to a folder QuantConnect custom data / Object Store can consume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_VERSION = 1


def _ohlcv_frame_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Build ``time,open,high,low,close,volume`` from Phi-nance normalized OHLCV."""
    if df.empty:
        raise ValueError("OHLCV dataframe is empty")
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex on OHLCV export")
    out = out.reset_index()
    idx_name = out.columns[0]
    out = out.rename(columns={idx_name: "time"})
    out["time"] = pd.to_datetime(out["time"]).dt.strftime("%Y-%m-%d")
    cols = {str(c).lower(): c for c in out.columns}
    for req in ("open", "high", "low", "close", "volume"):
        if req not in cols:
            raise ValueError(f"Missing column {req!r}; got {list(out.columns)}")
    return out[["time", "open", "high", "low", "close", "volume"]].copy()


def write_quantconnect_bundle(
    out_dir: Path | str,
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    ohlcv_vendor: str,
    df: pd.DataFrame,
    signal_card: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write ``ohlcv.csv``, ``manifest.json``, and optional JSON sidecars.

    Parameters
    ----------
    out_dir
        Directory created or reused for this export.
    symbol, timeframe, start, end, ohlcv_vendor
        Recorded in the manifest for reproducibility on QuantConnect.
    df
        Normalized OHLCV (DatetimeIndex, lowercase ohlcv columns).
    signal_card
        Serialized dict (e.g. ``OptionsSignalCard.model_dump()``).
    extras
        Arbitrary JSON-serializable blobs written as ``extras.json``.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    sym = symbol.strip().upper()

    csv_df = _ohlcv_frame_for_csv(df)
    ohlcv_path = root / "ohlcv.csv"
    csv_df.to_csv(ohlcv_path, index=False)

    files_meta: dict[str, Any] = {
        "ohlcv_csv": "ohlcv.csv",
        "columns": ["time", "open", "high", "low", "close", "volume"],
    }
    if signal_card is not None:
        (root / "signal_card.json").write_text(
            json.dumps(signal_card, indent=2, default=str) + "\n", encoding="utf-8"
        )
        files_meta["signal_card"] = "signal_card.json"
    if extras is not None:
        (root / "extras.json").write_text(
            json.dumps(extras, indent=2, default=str) + "\n", encoding="utf-8"
        )
        files_meta["extras"] = "extras.json"

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": sym,
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "ohlcv_vendor": ohlcv_vendor,
        "rows": int(len(csv_df)),
        "files": files_meta,
        "quantconnect_notes": {
            "resolution_hint": "Daily CSV maps to Resolution.Daily in Lean when ingested as custom data.",
            "custom_data_doc": "https://www.quantconnect.com/docs/v2/writing-algorithms/importing-data/streaming-data/custom-securities/key-concepts",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )

    return root

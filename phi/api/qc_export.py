"""Minimal FastAPI service: health check + trigger QuantConnect bundle export."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI(title="Phi-nance QuantConnect export", version="1.0.0")


def _export_root() -> Path:
    return Path(os.environ.get("PHINANCE_QC_EXPORT_DIR", "/tmp/qc_exports")).resolve()


class ExportResult(BaseModel):
    out_dir: str
    rows: int
    vendor: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/export/bundle", response_model=ExportResult)
def export_bundle(
    symbol: str = Query(..., min_length=1),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    timeframe: str = Query("1D"),
    include_signal_card: bool = Query(False),
    enrich_unusual_whales: bool = Query(True),
) -> ExportResult:
    """Run the same export as ``scripts/export_quantconnect_bundle.py`` into ``PHINANCE_QC_EXPORT_DIR``."""
    try:
        from dotenv import load_dotenv

        repo = Path(__file__).resolve().parents[2]
        load_dotenv(repo / ".env")
    except Exception:  # noqa: BLE001
        pass

    from phi.data.ohlcv_fallback import fetch_ohlcv_uw_then_yf
    from phi.integrations.quantconnect import write_quantconnect_bundle

    sym = symbol.strip().upper()
    root = _export_root()
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{sym}_{timeframe}_{start}_{end}".replace(":", "-")

    try:
        df, vendor = fetch_ohlcv_uw_then_yf(sym, start, end, timeframe=timeframe)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    signal_card: dict | None = None
    if include_signal_card:
        from phi.options.signal_generator import build_options_signal_card

        card = build_options_signal_card(
            df,
            symbol=sym,
            ohlcv_vendor=vendor,
            enrich_unusual_whales_chain=enrich_unusual_whales,
        )
        signal_card = card.model_dump(mode="json")

    write_quantconnect_bundle(
        out,
        symbol=sym,
        timeframe=timeframe,
        start=start,
        end=end,
        ohlcv_vendor=vendor,
        df=df,
        signal_card=signal_card,
    )
    return ExportResult(out_dir=str(out), rows=len(df), vendor=vendor)


class ExportBody(BaseModel):
    symbol: str = Field(..., min_length=1)
    start: str
    end: str
    timeframe: str = "1D"
    include_signal_card: bool = False
    enrich_unusual_whales: bool = True


@app.post("/export/bundle/json", response_model=ExportResult)
def export_bundle_json(body: ExportBody) -> ExportResult:
    return export_bundle(
        symbol=body.symbol,
        start=body.start,
        end=body.end,
        timeframe=body.timeframe,
        include_signal_card=body.include_signal_card,
        enrich_unusual_whales=body.enrich_unusual_whales,
    )

"""
Phase 6.5: Paper trading validation (8 weeks). Mandatory before any live capital.

Tradier sandbox runner emits daily ProjectionPacket for whole universe.
Kill criteria: AUC < 0.50 for 10 straight days; 75% cone coverage < 60% or > 90%; IC negative 2+ weeks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from phinence.contracts.projection_packet import ProjectionPacket


@dataclass
class PaperMetrics:
    """Rolling dashboard metrics."""
    window_days: int = 20
    auc: float = 0.0
    cone_75_coverage: float = 0.0
    ic: float = 0.0
    regime_dist: dict[str, float] = field(default_factory=dict)


PAPER_KILL_AUC_THRESHOLD = 0.50
PAPER_KILL_AUC_DAYS = 10
PAPER_KILL_CONE_LOW = 0.60
PAPER_KILL_CONE_HIGH = 0.90
PAPER_KILL_IC_WEEKS = 2


def should_kill_paper(
    metrics_history: list[PaperMetrics],
    auc_threshold: float = PAPER_KILL_AUC_THRESHOLD,
    auc_days: int = PAPER_KILL_AUC_DAYS,
    cone_low: float = PAPER_KILL_CONE_LOW,
    cone_high: float = PAPER_KILL_CONE_HIGH,
    ic_negative_weeks: int = PAPER_KILL_IC_WEEKS,
) -> tuple[bool, list[str]]:
    """
    Returns (should_kill, reasons).
    """
    reasons: list[str] = []
    if len(metrics_history) < auc_days:
        return False, []
    recent = metrics_history[-auc_days:]
    if all(m.auc < auc_threshold for m in recent):
        reasons.append(f"AUC < {auc_threshold} for {auc_days} straight days")
    last = metrics_history[-1] if metrics_history else None
    if last and (last.cone_75_coverage < cone_low or last.cone_75_coverage > cone_high):
        reasons.append(f"75% cone coverage {last.cone_75_coverage:.2%} outside [{cone_low:.0%}, {cone_high:.0%}]")
    weeks_negative = 0
    for m in reversed(metrics_history):
        if m.ic < 0:
            weeks_negative += 1
        else:
            break
    if weeks_negative >= ic_negative_weeks * 5:  # ~5 td per week
        reasons.append(f"IC negative for {ic_negative_weeks}+ weeks")
    return len(reasons) > 0, reasons


def paper_run_daily(
    tickers: list[str],
    bar_store: Any,
    assigner: Any,
    engines: dict[str, Any],
    composer: Any,
    as_of: datetime | None = None,
    train_start_ts: Any = None,
    horizon: str = "1d",
) -> list[ProjectionPacket]:
    """
    Emit one ProjectionPacket per ticker for the given as_of date.
    Uses bar_store for bars up to as_of; assign -> engines -> MFM -> composer.
    """
    from phinence.contracts.projection_packet import Horizon, make_stub_packet
    from phinence.mfm.merger import build_mfm

    import pandas as pd
    as_of = as_of or datetime.utcnow()
    if not hasattr(as_of, "hour"):
        as_of = datetime(as_of.year, as_of.month, as_of.day, 16, 0, 0) if hasattr(as_of, "year") else as_of
    end_ts = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
    start_ts = train_start_ts
    horizon_enum = Horizon.DAILY if horizon == "1d" else (Horizon.INTRADAY_5M if horizon == "5m" else Horizon.INTRADAY_1M)
    packets: list[ProjectionPacket] = []
    for ticker in tickers:
        try:
            df = bar_store.read_1m_bars(ticker) if hasattr(bar_store, "read_1m_bars") else None
            if df is None or df.empty or len(df) < 50:
                packets.append(make_stub_packet(ticker, as_of))
                continue
            ticker_start = start_ts
            if ticker_start is None and hasattr(df, "timestamp") and "timestamp" in df.columns:
                try:
                    ticker_start = pd.Timestamp(df["timestamp"].min())
                except Exception:
                    ticker_start = end_ts - pd.Timedelta(days=180)
            elif ticker_start is None:
                ticker_start = end_ts - pd.Timedelta(days=180)
            packet = assigner.assign(ticker, as_of, start_ts=ticker_start, end_ts=end_ts)
            liq = engines.get("liquidity")
            reg = engines.get("regime")
            sent = engines.get("sentiment")
            hed = engines.get("hedge")
            if liq:
                liq = liq.run(packet)
            if reg:
                reg = reg.run(packet)
            if sent:
                sent = sent.run(packet)
            if hed:
                hed = hed.run(packet)
            mfm = build_mfm(ticker, as_of, liquidity=liq, regime=reg, sentiment=sent, hedge=hed)
            proj = composer.run(mfm, horizons=[horizon_enum])
            packets.append(proj)
        except Exception:
            packets.append(make_stub_packet(ticker, as_of))
    return packets


def save_packets(packets: list[ProjectionPacket], out_dir: str | Path) -> list[Path]:
    """Write each packet to out_dir/{ticker}.json. Returns paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for p in packets:
        path = out / f"{p.ticker}.json"
        path.write_text(p.model_dump_json(), encoding="utf-8")
        paths.append(path)
    return paths


def load_packets_from_dir(date_dir: Path) -> list[ProjectionPacket]:
    """Load all ProjectionPacket JSONs from a date directory (e.g. data/paper_packets/2024-01-15)."""
    packets: list[ProjectionPacket] = []
    for f in date_dir.glob("*.json"):
        try:
            packets.append(ProjectionPacket.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return packets

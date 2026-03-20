"""Load options Greeks from Unusual Whales chain + summarize options flow for backtests."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from phi.exceptions import BacktestError
from phi.logging import get_logger
from phi.run_config import RunConfig

logger = get_logger(__name__)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def _is_call(v: Any) -> bool:
    s = str(v).lower().strip()
    return s in ("call", "c", "calls", "call option")


def _is_put(v: Any) -> bool:
    s = str(v).lower().strip()
    return s in ("put", "p", "puts", "put option")


def _pick_expiry_col(df: pd.DataFrame) -> str:
    for c in ("expiration", "expiry", "expiration_date", "exp_date"):
        if c in df.columns:
            return c
    raise BacktestError("Options chain has no expiration column (expected expiration / expiry).")


def _normalize_iv(raw: float) -> float:
    iv = float(raw)
    if iv > 1.5:
        iv = iv / 100.0
    return max(iv, 0.01)


def select_atm_chain_row(
    chain: pd.DataFrame,
    spot: float,
    want_call: bool,
    as_of: date,
) -> pd.Series:
    """Pick nearest expiry on/after *as_of*, then strike closest to *spot*."""
    if chain is None or chain.empty:
        raise BacktestError("Unusual Whales returned an empty options chain.")

    df = _norm_cols(chain)
    exp_col = _pick_expiry_col(df)

    if "strike" not in df.columns:
        raise BacktestError("Options chain missing strike column.")

    ot_col = next((c for c in ("option_type", "type", "cp_flag", "call_put") if c in df.columns), None)
    if ot_col is None:
        raise BacktestError("Options chain missing option type column.")

    df = df.copy()
    df["_exp"] = pd.to_datetime(df[exp_col], errors="coerce").dt.date
    df = df[df["_exp"].notna()]
    if df.empty:
        raise BacktestError("Could not parse option expiries from Unusual Whales chain.")

    side_mask = df[ot_col].map(_is_call if want_call else _is_put)
    df = df[side_mask]
    if df.empty:
        raise BacktestError(f"No {'call' if want_call else 'put'} rows in Unusual Whales chain.")

    future = sorted(e for e in df["_exp"].unique() if e >= as_of)
    chosen_exp = future[0] if future else max(df["_exp"].unique())
    sub = df[df["_exp"] == chosen_exp].copy()
    sub["_dist"] = (pd.to_numeric(sub["strike"], errors="coerce") - float(spot)).abs()
    sub = sub.sort_values("_dist")
    row = sub.iloc[0]
    return row


def build_uw_options_context(
    symbol: str,
    spot: float,
    want_call: bool,
    as_of: date,
) -> dict[str, Any]:
    """Fetch UW chain + flow; return overrides for RunConfig option_params and display context."""
    from phinance.data.vendors.unusual_whales import UnusualWhalesClient

    client = UnusualWhalesClient()
    chain = client.fetch_options_chain(symbol)
    row = select_atm_chain_row(chain, spot=spot, want_call=want_call, as_of=as_of)

    df = _norm_cols(chain)
    exp_col = _pick_expiry_col(df)
    if "_exp" in row.index and pd.notna(row["_exp"]):
        raw_e = row["_exp"]
        expiry_dt = raw_e if isinstance(raw_e, date) else pd.Timestamp(raw_e).date()
    else:
        exp_raw = row.get(exp_col) if exp_col in row.index else row.get("expiration")
        expiry_dt = pd.to_datetime(exp_raw, errors="coerce").date()
    try:
        bad = expiry_dt is None or pd.isna(expiry_dt)
    except (ValueError, TypeError):
        bad = True
    if bad:
        raise BacktestError("Could not parse expiry from selected chain row.")

    strike = float(pd.to_numeric(row.get("strike"), errors="coerce"))
    if strike <= 0:
        raise BacktestError("Invalid strike from Unusual Whales chain.")

    iv_col = next((c for c in ("implied_volatility", "iv", "implied_vol") if c in row.index), None)
    if iv_col is None:
        raise BacktestError("Chain row missing implied volatility (Greeks pricing requires IV).")
    iv = _normalize_iv(float(pd.to_numeric(row.get(iv_col), errors="coerce")))

    greeks: dict[str, float | None] = {}
    for g in ("delta", "gamma", "theta", "vega", "rho"):
        if g in row.index:
            try:
                greeks[g] = float(pd.to_numeric(row.get(g), errors="coerce"))
            except (TypeError, ValueError):
                greeks[g] = None
        else:
            greeks[g] = None

    flow_df = client.fetch_flow_alerts(symbol=symbol, limit=200)
    flow_summary = _summarize_flow_df(flow_df)

    overrides = {
        "strike": strike,
        "expiry": expiry_dt,
        "iv": iv,
    }

    chain_snapshot: dict[str, Any] = {
        "strike": strike,
        "expiration": str(expiry_dt),
        "implied_volatility": iv,
    }
    for g in ("delta", "gamma", "theta", "vega", "rho"):
        if g in row.index and pd.notna(row.get(g)):
            try:
                chain_snapshot[g] = float(pd.to_numeric(row.get(g), errors="coerce"))
            except (TypeError, ValueError):
                pass

    return {
        "overrides": overrides,
        "greeks_from_chain": greeks,
        "flow_summary": flow_summary,
        "chain_row": chain_snapshot,
    }


def _summarize_flow_df(flow_df: pd.DataFrame) -> dict[str, Any]:
    if flow_df is None or flow_df.empty:
        return {"alerts": 0, "note": "No flow alerts returned for this symbol."}

    df = _norm_cols(flow_df)
    out: dict[str, Any] = {"alerts": int(len(df))}

    prem_col = next((c for c in ("premium", "total_premium", "prem") if c in df.columns), None)
    if prem_col:
        prem = pd.to_numeric(df[prem_col], errors="coerce").fillna(0.0)
        out["premium_sum"] = float(prem.sum())

    sent_col = next((c for c in ("sentiment", "bias") if c in df.columns), None)
    if sent_col:
        vc = df[sent_col].astype(str).str.lower().value_counts().head(5)
        out["sentiment_counts"] = {str(k): int(v) for k, v in vc.items()}

    ot = next((c for c in ("option_type", "type") if c in df.columns), None)
    if ot:
        vc = df[ot].astype(str).str.lower().value_counts().head(6)
        out["option_type_counts"] = {str(k): int(v) for k, v in vc.items()}

    return out


def enrich_run_config_options_from_uw(config: RunConfig, ohlcv: pd.DataFrame) -> tuple[RunConfig, dict[str, Any]]:
    """Merge UW chain (Greeks/IV/strike/expiry) + flow summary; return updated config and context for UI."""
    if config.trading_mode != "options":
        return config, {}
    if len(config.symbols) != 1:
        raise BacktestError("Unusual Whales options enrichment supports a single symbol.")
    sym = config.symbols[0]
    base = config.option_params.get(sym)
    if not base:
        raise BacktestError("Missing option_params for symbol.")

    spot = float(ohlcv["close"].iloc[0])
    want_call = str(base.get("option_type", "call")).lower() == "call"
    ctx = build_uw_options_context(sym, spot=spot, want_call=want_call, as_of=config.start_date)

    merged = {**base, **ctx["overrides"]}
    # Preserve quantity, r, multiplier, style from form
    new_config = config.model_copy(update={"option_params": {sym: merged}})

    ui_ctx = {
        "symbol": sym,
        "spot_at_entry_bar": spot,
        "greeks_from_chain": ctx["greeks_from_chain"],
        "flow_summary": ctx["flow_summary"],
        "selected_chain_columns": ctx["chain_row"],
    }
    logger.info(
        "Unusual Whales options context: strike=%s expiry=%s iv=%.4f flow_alerts=%s",
        merged.get("strike"),
        merged.get("expiry"),
        float(merged.get("iv", 0)),
        ctx["flow_summary"].get("alerts"),
    )
    return new_config, ui_ctx

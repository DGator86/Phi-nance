"""Small time helpers for daily-bar context (US equity cash session)."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def seconds_until_next_us_equity_daily_close(now: datetime | None = None) -> tuple[int, str]:
    """Seconds until next weekday 4:00 PM America/New_York (rough regular-session close)."""
    et = ZoneInfo("America/New_York")
    now = now or datetime.now(et)
    if now.tzinfo is None:
        now = now.replace(tzinfo=et)
    else:
        now = now.astimezone(et)

    def next_close_from_date(d) -> datetime:
        while d.weekday() >= 5:
            d = d + timedelta(days=1)
        return datetime(d.year, d.month, d.day, 16, 0, 0, tzinfo=et)

    d = now.date()
    target = next_close_from_date(d)
    if now >= target or now.weekday() >= 5:
        d = d + timedelta(days=1)
        target = next_close_from_date(d)
    sec = max(0, int((target - now).total_seconds()))
    return sec, target.strftime("%Y-%m-%d %H:%M %Z")

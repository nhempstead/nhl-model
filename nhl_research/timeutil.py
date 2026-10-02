"""UTC-first time helpers. Display conversion is explicit and never used for joins."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
DISPLAY_TZ = ZoneInfo("America/New_York")


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Naive datetime rejected; supply an aware UTC timestamp")
    return value.astimezone(UTC)


def parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return ensure_utc(parsed)


def to_iso(value: datetime) -> str:
    return ensure_utc(value).isoformat().replace("+00:00", "Z")


def forecast_time(scheduled_start: datetime, horizon_minutes: int) -> datetime:
    start = ensure_utc(scheduled_start)
    if horizon_minutes < 0:
        raise ValueError("Forecast horizon must be non-negative")
    return start - timedelta(minutes=horizon_minutes)


def display_et(value: datetime) -> str:
    return ensure_utc(value).astimezone(DISPLAY_TZ).isoformat()

"""Canonical entities, enums, and forecast-record schema."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nhl_research.timeutil import ensure_utc


class DatasetOrigin(str, Enum):
    REAL = "REAL"
    REAL_API_SNAPSHOT = "REAL_API_SNAPSHOT"
    REAL_DERIVED = "REAL_DERIVED"
    SYNTHETIC = "SYNTHETIC"
    UNAVAILABLE = "UNAVAILABLE"


class SourceStatus(str, Enum):
    VERIFIED_AVAILABLE = "VERIFIED_AVAILABLE"
    REQUIRES_CREDENTIALS = "REQUIRES_CREDENTIALS"
    REQUIRES_PAID_ACCESS = "REQUIRES_PAID_ACCESS"
    UNVERIFIED = "UNVERIFIED"
    UNAVAILABLE = "UNAVAILABLE"


class ResearchStatus(str, Enum):
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    PASS = "PASS"
    BLOCKED = "BLOCKED"


class MarketType(str, Enum):
    MONEYLINE_FULL_GAME = "moneyline_full_game"
    TOTALS = "totals"
    PUCK_LINE = "puck_line"


class SettlementResult(str, Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    PUSH = "PUSH"
    VOID = "VOID"
    UNSETTLED = "UNSETTLED"
    UNAVAILABLE = "UNAVAILABLE"


class AwareModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _reject_naive_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return ensure_utc(value)
        return value


class Game(AwareModel):
    game_id: str
    season: int
    scheduled_start_utc: datetime | None = None
    scheduled_start_known_at: datetime | None = None
    start_time_precision: str = "unknown"
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    home_win: int | None = None
    game_state: str = "UNKNOWN"
    period_type: str | None = None
    venue: str | None = None
    source: str
    available_at: datetime
    ingested_at: datetime
    origin: DatasetOrigin
    quality_flags: list[str] = Field(default_factory=list)


class OddsSnapshot(AwareModel):
    snapshot_id: str
    game_id: str
    bookmaker: str
    market_type: MarketType
    home_american: int
    away_american: int
    commence_time_utc: datetime
    snapshot_time_utc: datetime
    ingested_at: datetime
    origin: DatasetOrigin
    source: str
    quality_flags: list[str] = Field(default_factory=list)


class ForecastRecord(AwareModel):
    game_id: str
    prediction_time: datetime
    market_contract: str
    selection: str
    model_version: str
    data_as_of: datetime
    probability: float | None
    interval_method: str | None
    interval_low: float | None
    interval_high: float | None
    interval_level: float | None = 0.95
    market_probability: float | None
    quote_american: int | None
    quote_decimal: float | None
    quote_time: datetime | None
    expected_return: float | None
    expected_return_low: float | None
    expected_return_high: float | None
    data_quality_flags: list[str] = Field(default_factory=list)
    research_status: ResearchStatus
    reason: str
    origin: DatasetOrigin
    settled_outcome: int | None = None

    def dump(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        return payload

"""Odds snapshot ingestion and point-in-time quote selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nhl_research.data.identity import odds_name_to_abbrev
from nhl_research.entities import DatasetOrigin
from nhl_research.markets.conversion import american_to_decimal, proportional_devig
from nhl_research.timeutil import forecast_time, parse_utc

UTC = timezone.utc


def load_local_odds_api_csv(path: Path, ingested_at: datetime | None = None) -> pd.DataFrame:
    ingested_at = ingested_at or datetime.now(tz=UTC)
    raw = pd.read_csv(path)
    rows = []
    for rec in raw.itertuples(index=False):
        home = odds_name_to_abbrev(str(rec.home_team))
        away = odds_name_to_abbrev(str(rec.away_team))
        flags = []
        if home is None or away is None:
            flags.append("UNMAPPED_TEAM_NAME")
        snapshot = parse_utc(str(rec.snapshot_time))
        commence = parse_utc(str(rec.commence_time))
        if snapshot >= commence:
            flags.append("BLOCK_SNAPSHOT_AT_OR_AFTER_START")
        rows.append(
            {
                "snapshot_id": f"{rec.game_id}:{rec.bookmaker}:{rec.snapshot_time}",
                "provider_game_id": rec.game_id,
                "home_team_name": rec.home_team,
                "away_team_name": rec.away_team,
                "home_team": home,
                "away_team": away,
                "bookmaker": rec.bookmaker,
                "market_type": "moneyline_full_game",
                "home_american": int(rec.home_odds),
                "away_american": int(rec.away_odds),
                "commence_time_utc": commence,
                "snapshot_time_utc": snapshot,
                "ingested_at": ingested_at,
                "origin": DatasetOrigin.REAL.value,
                "source": str(path),
                "quality_flags": "|".join(flags),
            }
        )
    return pd.DataFrame(rows)


def select_quote_at_horizon(
    odds: pd.DataFrame,
    *,
    horizon_minutes: int = 60,
    bookmaker: str | None = None,
) -> pd.DataFrame:
    """Latest snapshot at or before commence_time - horizon, same book, two-way market.

    Does not pick the best price in a window. Quotes at/after start are dropped.
    """
    if odds.empty:
        return odds
    work = odds.copy()
    if "provider_game_id" not in work.columns:
        work["provider_game_id"] = work["game_id"]
    if bookmaker:
        work = work[work["bookmaker"] == bookmaker]
    work = work[~work["quality_flags"].fillna("").str.contains("BLOCK_SNAPSHOT_AT_OR_AFTER_START")]
    if work.empty:
        return work
    cutoff = work["commence_time_utc"].map(lambda t: forecast_time(_as_dt(t), horizon_minutes))
    work = work.assign(prediction_time=cutoff)
    work = work[work["snapshot_time_utc"].map(_as_dt) <= work["prediction_time"]]
    if work.empty:
        return work
    work = work.sort_values(["provider_game_id", "snapshot_time_utc"])
    # One quote per event: latest eligible snapshot. Do not mix books into a best-price.
    # If multiple books share the same latest timestamp, keep a stable bookmaker order.
    work = work.sort_values(["provider_game_id", "snapshot_time_utc", "bookmaker"])
    latest = work.groupby(["provider_game_id"], as_index=False).tail(1)
    records = []
    for rec in latest.to_dict(orient="records"):
        try:
            home_dec = american_to_decimal(rec["home_american"])
            away_dec = american_to_decimal(rec["away_american"])
            devig = proportional_devig([home_dec, away_dec])
            market_home = devig.de_vigged[0]
            flags = rec.get("quality_flags") or ""
        except Exception as exc:  # noqa: BLE001
            home_dec = None
            away_dec = None
            market_home = None
            flags = (str(rec.get("quality_flags") or "") + "|" if rec.get("quality_flags") else "") + f"BLOCK_ODDS_INVALID:{exc}"
        age = (_as_dt(rec["prediction_time"]) - _as_dt(rec["snapshot_time_utc"])).total_seconds() / 60.0
        records.append(
            {
                "odds_game_id": rec["provider_game_id"],
                "home_team": rec.get("home_team"),
                "away_team": rec.get("away_team"),
                "bookmaker": rec["bookmaker"],
                "prediction_time": rec["prediction_time"],
                "quote_time": rec["snapshot_time_utc"],
                "commence_time_utc": rec["commence_time_utc"],
                "home_american": rec["home_american"],
                "away_american": rec["away_american"],
                "home_decimal": home_dec,
                "away_decimal": away_dec,
                "market_home_prob": market_home,
                "quote_age_minutes": age,
                "overround": None if home_dec is None else (1 / home_dec + 1 / away_dec - 1),
                "quality_flags": flags,
                "origin": rec.get("origin"),
            }
        )
    return pd.DataFrame(records)


def match_quotes_to_games(games: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Join quotes to games on date + teams. Unmatched games keep null quotes."""
    if games.empty:
        return games
    left = games.copy()
    if quotes is None or quotes.empty:
        left["market_home_prob"] = None
        left["home_decimal"] = None
        left["home_american"] = None
        left["quote_time"] = pd.NaT
        left["quote_age_minutes"] = None
        left["bookmaker"] = None
        left["odds_quality_flags"] = "BLOCK_NO_HISTORICAL_ODDS"
        return left
    q = quotes.copy()
    if "odds_game_id" in q.columns and set(left["game_id"]).intersection(set(q["odds_game_id"].astype(str))):
        merged = left.merge(q, how="left", left_on="game_id", right_on="odds_game_id", suffixes=("", "_odds"))
    else:
        left["match_date"] = pd.to_datetime(left["game_date"]).dt.date
        q["match_date"] = pd.to_datetime(q["commence_time_utc"], utc=True).dt.date
        merged = left.merge(
            q,
            how="left",
            left_on=["match_date", "home_team", "away_team"],
            right_on=["match_date", "home_team", "away_team"],
            suffixes=("", "_odds"),
        )
    merged["odds_quality_flags"] = merged.get("quality_flags", "")
    missing = merged["market_home_prob"].isna()
    merged.loc[missing, "odds_quality_flags"] = merged.loc[missing, "odds_quality_flags"].fillna(
        ""
    ).replace("", "BLOCK_NO_HISTORICAL_ODDS")
    return merged


def _as_dt(value) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if hasattr(value, "to_pydatetime"):
        return _as_dt(value.to_pydatetime())
    return parse_utc(str(value))

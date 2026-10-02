"""Ingestion orchestration with origin labels and future-game safeguards."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nhl_research.config import Paths
from nhl_research.data.moneypuck import canonical_games_from_moneypuck, load_local_all_teams
from nhl_research.data.nhl import fetch_score_date, games_from_score_payload
from nhl_research.data.odds import load_local_odds_api_csv, match_quotes_to_games, select_quote_at_horizon
from nhl_research.data.store import Warehouse
from nhl_research.data.synthetic import make_synthetic_league
from nhl_research.entities import DatasetOrigin
from nhl_research.exceptions import LeakageError

UTC = timezone.utc


def ingest_synthetic(warehouse: Warehouse, **kwargs) -> dict:
    games, odds = make_synthetic_league(**kwargs)
    warehouse.replace_table(
        "games",
        games,
        origin=DatasetOrigin.SYNTHETIC,
        source="nhl_research.data.synthetic",
        note="SYNTHETIC_OFFLINE_FIXTURE. Not real NHL performance.",
        primary_key=["game_id"],
    )
    warehouse.replace_table(
        "odds_snapshots",
        odds,
        origin=DatasetOrigin.SYNTHETIC,
        source="nhl_research.data.synthetic",
        note="SYNTHETIC_OFFLINE_FIXTURE quotes.",
        primary_key=["snapshot_id"],
    )
    return {"games": len(games), "odds_snapshots": len(odds), "origin": DatasetOrigin.SYNTHETIC.value}


def ingest_nhl_score_date(warehouse: Warehouse, game_date: str, as_of: datetime | None = None) -> dict:
    as_of = as_of or datetime.now(tz=UTC)
    payload = fetch_score_date(game_date)
    rows = games_from_score_payload(payload, ingested_at=as_of)
    frame = pd.DataFrame(rows)
    _reject_future_as_settled(frame, as_of)
    warehouse.write_table(
        "games",
        frame,
        origin=DatasetOrigin.REAL_API_SNAPSHOT,
        source=f"https://api-web.nhle.com/v1/score/{game_date}",
        note="NHL Web API score endpoint snapshot.",
        primary_key=["game_id"],
    )
    return {"games": len(frame), "origin": DatasetOrigin.REAL_API_SNAPSHOT.value, "date": game_date}


def ingest_nhl_score_fixture(warehouse: Warehouse, path: Path) -> dict:
    import json

    blob = json.loads(path.read_text(encoding="utf-8"))
    origin = DatasetOrigin(blob.get("dataset_origin", DatasetOrigin.REAL_API_SNAPSHOT.value))
    payload = blob["payload"]
    payload["_meta"] = {
        "endpoint": blob.get("endpoint"),
        "retrieved_at_utc": blob.get("retrieved_at_utc"),
        "origin": origin.value,
    }
    rows = games_from_score_payload(payload)
    frame = pd.DataFrame(rows)
    warehouse.write_table(
        "games",
        frame,
        origin=origin,
        source=str(path),
        note=blob.get("note", "fixture"),
        primary_key=["game_id"],
    )
    return {"games": len(frame), "origin": origin.value}


def ingest_moneypuck_local(warehouse: Warehouse, csv_path: Path) -> dict:
    raw = load_local_all_teams(csv_path)
    games = canonical_games_from_moneypuck(raw)
    warehouse.replace_table(
        "games",
        games,
        origin=DatasetOrigin.REAL_DERIVED,
        source=str(csv_path),
        note="MoneyPuck all-situation games. Tied scores have null moneyline labels.",
        primary_key=["game_id"],
    )
    return {"games": len(games), "origin": DatasetOrigin.REAL_DERIVED.value}


def ingest_featured_matchups(warehouse: Warehouse, csv_path: Path) -> dict:
    """Load the repository's precomputed matchup table with provenance flags.

    Rolling columns in this file are treated as REAL_DERIVED research features.
    Moneyline labels are null when listed goals are tied.
    """
    raw = pd.read_csv(csv_path)
    raw["game_id"] = raw["gameId"].astype(str)
    raw["season"] = raw["season"].astype(int)
    raw["game_date"] = pd.to_datetime(raw["gameDate"]).dt.date.astype(str)
    raw["home_team"] = raw["h_team"]
    raw["away_team"] = raw["a_team"]
    flags = []
    home_win = []
    for rec in raw.itertuples(index=False):
        tied = rec.h_totalGoalsFor == rec.h_totalGoalsAgainst
        if tied:
            flags.append("MONEYLINE_LABEL_UNAVAILABLE_TIED_GOALS|FEATURED_PRECOMPUTED")
            home_win.append(None)
        else:
            flags.append("FEATURED_PRECOMPUTED")
            home_win.append(int(rec.home_win))
    raw["home_win"] = home_win
    raw["quality_flags"] = flags
    raw["origin"] = DatasetOrigin.REAL_DERIVED.value
    raw["source"] = str(csv_path)
    raw["start_time_precision"] = "date_only"
    raw["scheduled_start_utc"] = pd.to_datetime(raw["game_date"] + "T00:00:00Z", utc=True)
    raw["available_at"] = raw["scheduled_start_utc"] + pd.Timedelta(hours=12)
    raw["ingested_at"] = datetime.now(tz=UTC)
    raw["home_goals"] = raw["h_totalGoalsFor"]
    raw["away_goals"] = raw["a_totalGoalsFor"]
    raw["game_state"] = "OFF"
    keep_prefix = (
        "game_id",
        "season",
        "game_date",
        "home_team",
        "away_team",
        "home_win",
        "home_goals",
        "away_goals",
        "scheduled_start_utc",
        "available_at",
        "ingested_at",
        "origin",
        "source",
        "quality_flags",
        "start_time_precision",
        "game_state",
    )
    extra = [c for c in raw.columns if c.endswith(("_L10", "_L20", "_L40", "_diff")) or c in {"rest_diff", "a_rest_days", "h_rest_days"}]
    games = raw[list(keep_prefix) + extra]
    warehouse.replace_table(
        "games",
        games,
        origin=DatasetOrigin.REAL_DERIVED,
        source=str(csv_path),
        note="Precomputed matchups from repository LFS. Rolling stats claimed shift(1) in legacy script.",
        primary_key=["game_id"],
    )
    return {"games": len(games), "origin": DatasetOrigin.REAL_DERIVED.value}


def ingest_odds_local(warehouse: Warehouse, csv_path: Path, horizon_minutes: int = 60) -> dict:
    snapshots = load_local_odds_api_csv(csv_path)
    warehouse.replace_table(
        "odds_snapshots",
        snapshots,
        origin=DatasetOrigin.REAL,
        source=str(csv_path),
        note="Local Odds-API-derived snapshots. One row per event in this file; not a full 5/10-minute history.",
        primary_key=["snapshot_id"],
    )
    quotes = select_quote_at_horizon(snapshots, horizon_minutes=horizon_minutes)
    warehouse.replace_table(
        "quotes_tminus",
        quotes,
        origin=DatasetOrigin.REAL,
        source=str(csv_path),
        note=f"Latest same-book snapshot at or before T-{horizon_minutes} minutes.",
        primary_key=["odds_game_id"],
    )
    return {"snapshots": len(snapshots), "quotes": len(quotes)}


def attach_quotes(warehouse: Warehouse) -> int:
    games = warehouse.read_table("games")
    quotes = warehouse.read_table("quotes_tminus")
    if quotes.empty:
        quotes = warehouse.read_table("odds_snapshots")
        if not quotes.empty:
            quotes = select_quote_at_horizon(quotes)
    merged = match_quotes_to_games(games, quotes)
    origin = DatasetOrigin(str(games["origin"].iloc[0])) if len(games) else DatasetOrigin.UNAVAILABLE
    warehouse.replace_table(
        "games_with_quotes",
        merged,
        origin=origin,
        source="join(games, quotes_tminus)",
        note="Left join; missing quotes remain null.",
        primary_key=["game_id"] if "game_id" in merged.columns else list(merged.columns[:1]),
    )
    return int(merged["market_home_prob"].notna().sum()) if "market_home_prob" in merged.columns else 0


def _reject_future_as_settled(frame: pd.DataFrame, as_of: datetime) -> None:
    if frame.empty:
        return
    settled = frame["home_win"].notna()
    future_flag = frame["quality_flags"].fillna("").str.contains("FUTURE_GAME")
    if bool((settled & future_flag).any()):
        raise LeakageError("Future-dated games cannot carry settled labels")
    starts = pd.to_datetime(frame["scheduled_start_utc"], utc=True)
    illegal = settled & (starts > pd.Timestamp(as_of))
    if bool(illegal.any()):
        raise LeakageError("Settled labels on games scheduled after as_of")

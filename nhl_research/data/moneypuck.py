"""MoneyPuck listed CSV downloads. Terms: non-commercial, credit required."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nhl_research.data.identity import normalize_team
from nhl_research.entities import DatasetOrigin
from nhl_research.timeutil import parse_utc

UTC = timezone.utc

# Documented on https://moneypuck.com/data.htm (verified 2026-10-02)
ALL_TEAMS_GAME_URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
SEASON_TEAMS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"

# Conservative publication lag when only a calendar date is known.
GAME_AVAILABLE_LAG_HOURS = 12


def load_local_all_teams(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        return frame
    return frame


def canonical_games_from_moneypuck(raw: pd.DataFrame, ingested_at: datetime | None = None) -> pd.DataFrame:
    """Collapse situation-level MoneyPuck rows to one game with a moneyline label.

    Labels are null when 'all' situation goals are tied: shootout winners are not
    reliably recoverable from this file. xG columns are provider model outputs;
    historical model version is not published, so xG is flagged.
    """
    ingested_at = ingested_at or datetime.now(tz=UTC)
    if raw.empty:
        return pd.DataFrame()
    sit = raw[raw["situation"] == "all"].copy()
    sit["team_norm"] = sit["team"].map(lambda x: normalize_team(str(x)))
    sit["opp_norm"] = sit["opposingTeam"].map(lambda x: normalize_team(str(x)))
    sit["is_home"] = sit["home_or_away"].str.upper().eq("HOME")
    home = sit[sit["is_home"]].copy()
    away = sit[~sit["is_home"]].copy()
    merged = home.merge(
        away,
        on="gameId",
        suffixes=("_h", "_a"),
        how="inner",
    )
    rows = []
    for rec in merged.itertuples(index=False):
        game_date = _parse_mp_date(getattr(rec, "gameDate_h"))
        # Date-only start: do not pretend we knew a faceoff time.
        scheduled = datetime(game_date.year, game_date.month, game_date.day, tzinfo=UTC)
        home_goals = int(getattr(rec, "goalsFor_h"))
        away_goals = int(getattr(rec, "goalsFor_a"))
        flags = ["MONEYPUCK_DATE_ONLY_START", "XG_PROVIDER_MODEL_VERSION_UNKNOWN"]
        home_win = None
        if home_goals == away_goals:
            flags.append("MONEYLINE_LABEL_UNAVAILABLE_TIED_GOALS")
        else:
            home_win = int(home_goals > away_goals)
        available = scheduled + timedelta(hours=GAME_AVAILABLE_LAG_HOURS)
        rows.append(
            {
                "game_id": str(int(getattr(rec, "gameId"))),
                "season": int(getattr(rec, "season_h")),
                "scheduled_start_utc": scheduled,
                "scheduled_start_known_at": scheduled,
                "start_time_precision": "date_only",
                "home_team": getattr(rec, "team_norm_h"),
                "away_team": getattr(rec, "team_norm_a"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_win": home_win,
                "game_state": "OFF" if home_win is not None else "OFF_UNSETTLED_SO",
                "source": "moneypuck_all_teams",
                "available_at": available,
                "ingested_at": ingested_at,
                "origin": DatasetOrigin.REAL_DERIVED.value,
                "quality_flags": "|".join(flags),
                "game_date": game_date.isoformat(),
                "h_xgoals": float(getattr(rec, "xGoalsFor_h")),
                "a_xgoals": float(getattr(rec, "xGoalsFor_a")),
                "h_xgoals_against": float(getattr(rec, "xGoalsAgainst_h")),
                "a_xgoals_against": float(getattr(rec, "xGoalsAgainst_a")),
                "h_corsi_pct": float(getattr(rec, "corsiPercentage_h")),
                "a_corsi_pct": float(getattr(rec, "corsiPercentage_a")),
            }
        )
    return pd.DataFrame(rows)


def team_game_stats_5on5(raw: pd.DataFrame) -> pd.DataFrame:
    sit = raw[raw["situation"] == "5on5"].copy()
    sit["team_norm"] = sit["team"].map(lambda x: normalize_team(str(x)))
    sit["game_date"] = sit["gameDate"].map(lambda x: _parse_mp_date(x).isoformat())
    sit["game_id"] = sit["gameId"].astype(str)
    keep = [
        "game_id",
        "season",
        "team_norm",
        "game_date",
        "xGoalsFor",
        "xGoalsAgainst",
        "goalsFor",
        "goalsAgainst",
        "corsiPercentage",
        "fenwickPercentage",
        "shotAttemptsFor",
        "shotAttemptsAgainst",
        "highDangerShotsFor",
        "highDangerShotsAgainst",
        "iceTime",
    ]
    present = [c for c in keep if c in sit.columns]
    out = sit[present].rename(columns={"team_norm": "team"})
    out["origin"] = DatasetOrigin.REAL_DERIVED.value
    out["quality_flags"] = "XG_PROVIDER_MODEL_VERSION_UNKNOWN|FIVE_ON_FIVE_ONLY"
    return out


def _parse_mp_date(value) -> datetime.date:
    text = str(int(value)) if not isinstance(value, str) else value.replace("-", "")
    return datetime.strptime(text[:8], "%Y%m%d").date()

"""Lightweight schema checks used at ingest time."""

from __future__ import annotations

import pandas as pd

from nhl_research.exceptions import LeakageError


REQUIRED_GAME_COLUMNS = {
    "game_id",
    "season",
    "home_team",
    "away_team",
    "origin",
}


def validate_games(frame: pd.DataFrame, as_of=None) -> None:
    missing = REQUIRED_GAME_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"games table missing {missing}")
    if as_of is not None and "scheduled_start_utc" in frame.columns and "home_win" in frame.columns:
        starts = pd.to_datetime(frame["scheduled_start_utc"], utc=True)
        settled = frame["home_win"].notna()
        if bool((settled & (starts > pd.Timestamp(as_of))).any()):
            raise LeakageError("future games labeled settled")

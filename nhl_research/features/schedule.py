"""Rest days and back-to-back flags from the known schedule only."""

from __future__ import annotations

import pandas as pd

from nhl_research.exceptions import LeakageError


def add_schedule_features(games: pd.DataFrame) -> pd.DataFrame:
    frame = games.sort_values(["scheduled_start_utc", "game_id"]).copy()
    last_start: dict[str, pd.Timestamp] = {}
    rest_home = []
    rest_away = []
    b2b_home = []
    b2b_away = []
    for rec in frame.itertuples(index=False):
        start = pd.Timestamp(rec.scheduled_start_utc)
        rh = _rest(last_start.get(rec.home_team), start)
        ra = _rest(last_start.get(rec.away_team), start)
        rest_home.append(rh)
        rest_away.append(ra)
        b2b_home.append(bool(rh is not None and rh <= 1))
        b2b_away.append(bool(ra is not None and ra <= 1))
        last_start[rec.home_team] = start
        last_start[rec.away_team] = start
    frame["h_rest_days"] = rest_home
    frame["a_rest_days"] = rest_away
    frame["rest_diff"] = [
        (h - a) if h is not None and a is not None else None for h, a in zip(rest_home, rest_away)
    ]
    frame["h_b2b"] = b2b_home
    frame["a_b2b"] = b2b_away
    return frame


def _rest(previous: pd.Timestamp | None, current: pd.Timestamp) -> float | None:
    if previous is None:
        return None
    delta = (current - previous).total_seconds() / 86400.0
    if delta < 0:
        raise LeakageError("Schedule not chronological for rest calculation")
    return float(delta)

"""Chronological Elo ratings with pregame values only."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from nhl_research.exceptions import LeakageError

DEFAULT_RATING = 1500.0
K_FACTOR = 6.0
HOME_ADVANTAGE = 50.0
SCALE = 400.0


def elo_win_probability(home_rating: float, away_rating: float, home_advantage: float = HOME_ADVANTAGE) -> float:
    diff = home_rating + home_advantage - away_rating
    return 1.0 / (1.0 + 10 ** (-diff / SCALE))


def add_pregame_elo(
    games: pd.DataFrame,
    *,
    k: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    as_of: datetime | None = None,
) -> pd.DataFrame:
    frame = games.sort_values(["scheduled_start_utc", "game_id"]).copy()
    ratings: dict[str, float] = {}
    n_prior: dict[str, int] = {}
    home_pre = []
    away_pre = []
    home_n = []
    away_n = []
    elo_p = []
    for rec in frame.itertuples(index=False):
        if as_of is not None:
            start = rec.scheduled_start_utc
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            if start_ts.to_pydatetime() > as_of:
                # Still produce pregame ratings, but do not update from this game.
                pass
        h = rec.home_team
        a = rec.away_team
        rh = ratings.get(h, DEFAULT_RATING)
        ra = ratings.get(a, DEFAULT_RATING)
        home_pre.append(rh)
        away_pre.append(ra)
        home_n.append(n_prior.get(h, 0))
        away_n.append(n_prior.get(a, 0))
        elo_p.append(elo_win_probability(rh, ra, home_advantage))
        y = rec.home_win
        if y is None or (isinstance(y, (float, np.floating)) and not np.isfinite(float(y))):
            continue
        if pd.isna(y):
            continue
        if as_of is None or pd.Timestamp(rec.scheduled_start_utc, tz="UTC") <= pd.Timestamp(as_of):
            expected = elo_win_probability(rh, ra, home_advantage)
            ratings[h] = rh + k * (float(y) - expected)
            ratings[a] = ra + k * ((1.0 - float(y)) - (1.0 - expected))
            n_prior[h] = n_prior.get(h, 0) + 1
            n_prior[a] = n_prior.get(a, 0) + 1
    frame["home_elo_pre"] = home_pre
    frame["away_elo_pre"] = away_pre
    frame["home_prior_games"] = home_n
    frame["away_prior_games"] = away_n
    frame["elo_diff"] = frame["home_elo_pre"] - frame["away_elo_pre"]
    frame["elo_home_win_prob"] = elo_p
    # Diagnostic: first game for a team must use the prior, not 1500 after an update from itself.
    return frame


def assert_elo_ignores_current_outcome(games: pd.DataFrame) -> None:
    if games.empty:
        return
    # Reconstruct: if we accidentally updated before storing pregame, home_elo_pre would
    # differ across a team's first two games inconsistently. Check the first row uses 1500.
    first = games.sort_values(["scheduled_start_utc", "game_id"]).iloc[0]
    if first["home_elo_pre"] != DEFAULT_RATING or first["away_elo_pre"] != DEFAULT_RATING:
        raise LeakageError("First-game Elo is not the default prior")

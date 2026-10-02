"""Point-in-time rolling statistics. Current-game outcomes are excluded."""

from __future__ import annotations

import pandas as pd

from nhl_research.exceptions import LeakageError


def rolling_pregame_mean(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    if window < 1:
        raise ValueError("window must be >= 1")
    periods = min_periods or max(3, window // 2)
    return series.shift(1).rolling(window=window, min_periods=periods).mean()


def assert_no_current_game_leak(raw: pd.Series, rolled: pd.Series) -> None:
    aligned = pd.concat({"raw": raw, "rolled": rolled}, axis=1).dropna()
    if aligned.empty or len(aligned) <= 3:
        return
    if bool((aligned["raw"] == aligned["rolled"]).all()):
        raise LeakageError("Rolling feature matches current-game values; shift(1) missing")


def build_team_rolling(
    team_games: pd.DataFrame,
    value_cols: list[str],
    windows: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    required = {"team", "game_id", "game_date"}
    missing = required - set(team_games.columns)
    if missing:
        raise ValueError(f"team_games missing {missing}")
    frame = team_games.sort_values(["team", "game_date", "game_id"]).copy()
    pieces = []
    for _, grp in frame.groupby("team", sort=False):
        local = grp.copy()
        for col in value_cols:
            if col not in local.columns:
                continue
            for window in windows:
                name = f"{col}_L{window}"
                local[name] = rolling_pregame_mean(local[col], window)
                assert_no_current_game_leak(local[col], local[name])
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True) if pieces else frame

"""Assemble the pregame feature matrix used by core models."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from nhl_research.features.ratings import add_pregame_elo
from nhl_research.features.schedule import add_schedule_features

CORE_MODEL_FEATURES = [
    "elo_diff",
    "rest_diff",
    "rest_missing",
    "h_prior_games_min",
    "xg_pct_L20_diff",
    "xg_pct_missing",
]


def assemble_features(games: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
    frame = add_schedule_features(games)
    frame = add_pregame_elo(frame, as_of=as_of)
    frame["h_prior_games_min"] = np.minimum(frame["home_prior_games"], frame["away_prior_games"])
    if "h_xGoalsPercentage_L20" in frame.columns and "a_xGoalsPercentage_L20" in frame.columns:
        frame["xg_pct_L20_diff"] = frame["h_xGoalsPercentage_L20"] - frame["a_xGoalsPercentage_L20"]
    elif "xGoalsPercentage_L20_diff" in frame.columns:
        frame["xg_pct_L20_diff"] = frame["xGoalsPercentage_L20_diff"]
    else:
        frame["xg_pct_L20_diff"] = np.nan
    # Unknown is not zero.
    frame["xg_pct_missing"] = frame["xg_pct_L20_diff"].isna().astype(int)
    frame["rest_missing"] = frame["rest_diff"].isna().astype(int)
    frame["low_exposure"] = (frame["h_prior_games_min"] < 10).astype(int)
    return frame


def model_matrix(frame: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    names = feature_names or CORE_MODEL_FEATURES
    mat = frame.copy()
    for name in names:
        if name not in mat.columns:
            mat[name] = np.nan
        if name == "rest_diff":
            # Median fill only inside a caller-provided training mask; here leave NaN.
            continue
        if name == "xg_pct_L20_diff":
            continue
    return mat

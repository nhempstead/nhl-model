"""Transparent baselines: historical base rate, Elo, and de-vigged market."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhl_research.features.ratings import elo_win_probability
from nhl_research.uncertainty.intervals import clip_probability_interval


@dataclass
class BaselinePrediction:
    name: str
    probability: np.ndarray
    interval_low: np.ndarray
    interval_high: np.ndarray
    interval_method: str
    notes: str


def expanding_home_base_rate(games: pd.DataFrame) -> BaselinePrediction:
    """League home-win rate using only previously settled games."""
    frame = games.sort_values(["scheduled_start_utc", "game_id"]).copy()
    wins = 0.0
    n = 0.0
    probs = []
    lows = []
    highs = []
    prior_p = 0.54
    for rec in frame.itertuples(index=False):
        alpha = 1.0 + wins
        beta = 1.0 + (n - wins)
        p = alpha / (alpha + beta) if n > 0 else prior_p
        # Jeffreys/Beta(1,1) 95% central interval for the *rate*, not the game.
        low = _beta_quantile(0.025, alpha, beta) if n >= 2 else 0.3
        high = _beta_quantile(0.975, alpha, beta) if n >= 2 else 0.7
        low, high = clip_probability_interval(low, high)
        probs.append(p)
        lows.append(low)
        highs.append(high)
        y = rec.home_win
        if y is None or pd.isna(y):
            continue
        wins += float(y)
        n += 1.0
    return BaselinePrediction(
        name="base_rate_expanding",
        probability=np.asarray(probs, dtype=float),
        interval_low=np.asarray(lows, dtype=float),
        interval_high=np.asarray(highs, dtype=float),
        interval_method="beta_binomial_posterior_for_league_rate",
        notes="Interval is uncertainty about the league home-win rate, not game-specific skill.",
    )


def elo_baseline(games: pd.DataFrame) -> BaselinePrediction:
    p = games["elo_home_win_prob"].to_numpy(dtype=float)
    # Rating SE approximation: 400 / sqrt(n_prior); map through the logistic.
    se_h = 400.0 / np.sqrt(np.maximum(games["home_prior_games"].to_numpy(dtype=float), 1.0))
    se_a = 400.0 / np.sqrt(np.maximum(games["away_prior_games"].to_numpy(dtype=float), 1.0))
    se_diff = np.sqrt(se_h**2 + se_a**2)
    home = games["home_elo_pre"].to_numpy(dtype=float)
    away = games["away_elo_pre"].to_numpy(dtype=float)
    low = np.array([elo_win_probability(h - 1.96 * s, a) for h, a, s in zip(home, away, se_diff)])
    high = np.array([elo_win_probability(h + 1.96 * s, a) for h, a, s in zip(home, away, se_diff)])
    # When h-se is used, also widen away. Conservative envelope:
    low2 = np.array([elo_win_probability(h, a + 1.96 * s) for h, a, s in zip(home, away, se_diff)])
    high2 = np.array([elo_win_probability(h, a - 1.96 * s) for h, a, s in zip(home, away, se_diff)])
    low = np.minimum(low, low2)
    high = np.maximum(high, high2)
    return BaselinePrediction(
        name="elo_pregame",
        probability=p,
        interval_low=low,
        interval_high=high,
        interval_method="elo_rating_se_delta_map",
        notes="Approximate parameter uncertainty from prior-game counts. Not a binomial SE of p.",
    )


def market_baseline(games: pd.DataFrame) -> BaselinePrediction:
    if "market_home_prob" not in games.columns:
        p = np.full(len(games), np.nan)
    else:
        p = pd.to_numeric(games["market_home_prob"], errors="coerce").to_numpy(dtype=float)
    missing = ~np.isfinite(p)
    interval_low = p.copy()
    interval_high = p.copy()
    # Market baseline has no model interval; interval is null where missing, else a placeholder nan.
    interval_low[missing] = np.nan
    interval_high[missing] = np.nan
    return BaselinePrediction(
        name="market_devig_same_book",
        probability=p,
        interval_low=interval_low,
        interval_high=interval_high,
        interval_method="none_quote_is_point_estimate",
        notes="De-vigged quote is a convention, not a posterior. Missing quotes stay null.",
    )


def _beta_quantile(p: float, a: float, b: float) -> float:
    # Use numpy's incomplete-beta inversion via a simple Newton on the regularized beta
    # Fallback: normal approximation for moderate a,b.
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    sd = np.sqrt(max(var, 1e-12))
    from nhl_research.uncertainty.intervals import _normal_ppf

    z = _normal_ppf(p)
    return float(np.clip(mean + z * sd, 0.0, 1.0))

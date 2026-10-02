"""Clearly labeled synthetic fixtures for software tests, not real performance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from nhl_research.entities import DatasetOrigin
from nhl_research.markets.conversion import american_to_decimal, proportional_devig

UTC = timezone.utc
TEAMS = ("AAA", "BBB", "CCC", "DDD")
STRENGTH = {"AAA": 0.12, "BBB": 0.04, "CCC": -0.03, "DDD": -0.13}


def make_synthetic_league(
    *,
    n_games: int = 80,
    seed: int = 20261002,
    start: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a 4-team league with a known home-advantage process.

    Origin is always SYNTHETIC. Metrics computed on this data demonstrate
    software behavior, not NHL forecasting skill.
    """
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 10, 1, 23, 0, tzinfo=UTC)
    games: list[dict] = []
    odds: list[dict] = []
    season = 2024
    for i in range(n_games):
        if i == n_games // 2:
            season = 2025
        home, away = rng.choice(TEAMS, size=2, replace=False)
        scheduled = start + timedelta(days=i // 2, hours=3 * (i % 2))
        logit = STRENGTH[home] - STRENGTH[away] + 0.16
        p_true = float(1.0 / (1.0 + np.exp(-logit)))
        y = int(rng.random() < p_true)
        home_goals = 3 + y + int(rng.integers(0, 2))
        away_goals = 3 + (1 - y) + int(rng.integers(0, 2))
        if home_goals == away_goals:
            home_goals += 1 if y else 0
            away_goals += 0 if y else 1
        game_id = f"SYN-{season}-{i:04d}"
        available = scheduled + timedelta(hours=3)
        games.append(
            {
                "game_id": game_id,
                "season": season,
                "scheduled_start_utc": scheduled,
                "scheduled_start_known_at": scheduled - timedelta(days=30),
                "start_time_precision": "exact",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_win": y,
                "game_state": "OFF",
                "source": "synthetic_generator",
                "available_at": available,
                "ingested_at": available,
                "origin": DatasetOrigin.SYNTHETIC.value,
                "quality_flags": "SYNTHETIC_OFFLINE_FIXTURE",
                "p_true": p_true,
                "game_date": scheduled.date().isoformat(),
            }
        )
        # Market is noisy true probability plus vig, quoted before the cutoff.
        p_mkt = float(np.clip(p_true + rng.normal(0, 0.03), 0.2, 0.8))
        snapshot = scheduled - timedelta(minutes=75)
        home_dec, away_dec = _prices_from_p(p_mkt, vig=0.04)
        odds.append(
            _odds_row(game_id, scheduled, snapshot, home_dec, away_dec, "synthbook", home, away)
        )
        if i == 0:
            # Illegal post-start snapshot for tests.
            odds.append(
                _odds_row(
                    game_id,
                    scheduled,
                    scheduled + timedelta(minutes=5),
                    home_dec,
                    away_dec,
                    "synthbook-late",
                    home,
                    away,
                )
            )
        if i % 11 == 0:
            # Some games have no usable quote (omit extra). The main quote remains.
            pass
        if i % 17 == 0:
            # Drop the on-time quote to create missing-odds games.
            odds.pop()
    games_df = pd.DataFrame(games)
    odds_df = pd.DataFrame(odds)
    return games_df, odds_df


def _prices_from_p(p_home: float, vig: float) -> tuple[float, float]:
    q_home = p_home * (1.0 + vig)
    q_away = (1.0 - p_home) * (1.0 + vig)
    return 1.0 / q_home, 1.0 / q_away


def _odds_row(
    game_id: str,
    commence: datetime,
    snapshot: datetime,
    home_dec: float,
    away_dec: float,
    book: str,
    home_team: str = "",
    away_team: str = "",
) -> dict:
    return {
        "snapshot_id": f"{game_id}:{book}:{snapshot.isoformat()}",
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "bookmaker": book,
        "market_type": "moneyline_full_game",
        "home_american": _dec_to_american(home_dec),
        "away_american": _dec_to_american(away_dec),
        "commence_time_utc": commence,
        "snapshot_time_utc": snapshot,
        "ingested_at": snapshot,
        "origin": DatasetOrigin.SYNTHETIC.value,
        "source": "synthetic_generator",
        "quality_flags": "SYNTHETIC_OFFLINE_FIXTURE",
    }


def _dec_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def assert_devig_example() -> None:
    """Sanity helper used by tests; not a real-market claim."""
    result = proportional_devig([american_to_decimal(-110), american_to_decimal(-110)])
    assert abs(sum(result.de_vigged) - 1.0) < 1e-12

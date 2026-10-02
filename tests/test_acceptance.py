"""Acceptance tests that fail if leakage, settlement, or promotion safeguards break."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from nhl_research.config import Paths
from nhl_research.data.ingest import ingest_synthetic
from nhl_research.data.odds import select_quote_at_horizon
from nhl_research.data.store import Warehouse
from nhl_research.data.synthetic import make_synthetic_league
from nhl_research.entities import DatasetOrigin, ResearchStatus
from nhl_research.evaluation.replay import build_forecast_frame, freeze_forecasts
from nhl_research.evaluation.walkforward import walk_forward
from nhl_research.exceptions import InvalidOddsError, LeakageError
from nhl_research.features.assemble import assemble_features
from nhl_research.features.ratings import DEFAULT_RATING
from nhl_research.features.rolling import rolling_pregame_mean
from nhl_research.ledger.paper import decisions_from_forecasts, ledger_summary, settle_ledger
from nhl_research.learning.lifecycle import evaluate_challenger, reject_automatic_win_streak
from nhl_research.markets.conversion import (
    american_to_decimal,
    expected_net_return,
    proportional_devig,
)
from nhl_research.markets.eligibility import evaluate_research_eligibility
from nhl_research.markets.settlement import settle_moneyline_full_game
from nhl_research.models.logistic import assert_scaler_not_fit_on_holdout, fit_logistic
from nhl_research.models.registry import ModelRegistry, ModelVersion
from nhl_research.uncertainty.intervals import margin_of_error, standard_error_iid_mean
from nhl_research.uncertainty.power import two_sample_paired_mean_power
from nhl_research.uncertainty.resampling import paired_block_bootstrap

UTC = timezone.utc


@pytest.fixture
def tmp_paths(tmp_path) -> Paths:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "nhl_research").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='t'\n")
    warehouse = root / "data" / "warehouse"
    reports = root / "reports"
    registry = root / "models" / "registry"
    warehouse.mkdir(parents=True)
    reports.mkdir()
    (registry / "versions").mkdir(parents=True)
    return Paths(
        root=root,
        warehouse=warehouse,
        reports=reports,
        registry=registry,
        raw=root / "data" / "raw",
        processed=root / "data" / "processed",
    )


def test_future_dated_inputs_rejected_and_rolling_excludes_current_game():
    values = pd.Series([1.0, 2.0, 99.0, 4.0])
    rolled = rolling_pregame_mean(values, window=2, min_periods=1)
    assert pd.isna(rolled.iloc[0])
    assert rolled.iloc[2] != 99.0
    games, _ = make_synthetic_league(n_games=20)
    as_of = datetime(2024, 10, 5, tzinfo=UTC)
    future = games.copy()
    future.loc[future.index[-1], "home_win"] = 1
    future.loc[future.index[-1], "scheduled_start_utc"] = datetime(2026, 12, 1, tzinfo=UTC)
    from nhl_research.data.validation import validate_games

    with pytest.raises(LeakageError):
        validate_games(future, as_of=as_of)


def test_preprocessing_not_fit_on_holdout():
    games, _ = make_synthetic_league(n_games=60)
    feat = assemble_features(games)
    train = feat[feat["season"] == 2024]
    test = feat[feat["season"] == 2025]
    art = fit_logistic(train, ["elo_diff", "rest_missing"], val=None, calibrate=False)
    assert_scaler_not_fit_on_holdout(art, test)
    # Fitting scaler statistics must come from train: mean of train elo_diff.
    train_mean = train["elo_diff"].mean()
    assert art.scaler.mean_[0] == pytest.approx(train_mean, rel=1e-6)


def test_related_forecasts_grouped_in_uncertainty():
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "game_date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "a": [0.1, 0.1, 0.2, 0.2],
            "b": [0.2, 0.2, 0.1, 0.1],
        }
    )
    report = paired_block_bootstrap(frame, "a", "b", n_replicates=50, seed=1)
    assert report.n_observations == 4
    assert report.n_unique_games == 2
    assert report.effective_sample_size == 2
    assert report.simulation_draws == 50
    assert report.simulation_draws_are_not_observations is True
    assert report.simulation_draws != report.n_unique_games


def test_duplicate_ingestion_idempotent(tmp_paths):
    warehouse = Warehouse(tmp_paths)
    ingest_synthetic(warehouse, n_games=12, seed=1)
    n1 = len(warehouse.read_table("games"))
    ingest_synthetic(warehouse, n_games=12, seed=1)
    n2 = len(warehouse.read_table("games"))
    assert n1 == n2 == 12


def test_missing_availability_not_treated_as_confirmed():
    games, _ = make_synthetic_league(n_games=8)
    feat = assemble_features(games)
    first = feat.sort_values("scheduled_start_utc").iloc[0]
    assert pd.isna(first["rest_diff"]) or first["rest_missing"] == 1
    assert first["home_elo_pre"] == DEFAULT_RATING
    # Unknown xG remains missing, not zeroed in the assembled column.
    assert bool(feat["xg_pct_L20_diff"].isna().all())


def test_odds_math_devig_returns_push_void():
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    with pytest.raises(InvalidOddsError):
        american_to_decimal(0)
    with pytest.raises(InvalidOddsError):
        american_to_decimal("abc")
    dv = proportional_devig([american_to_decimal(-110), american_to_decimal(-110)])
    assert dv.de_vigged[0] == pytest.approx(0.5)
    assert abs(sum(dv.de_vigged) - 1.0) < 1e-12
    assert expected_net_return(0.55, 1.91) == pytest.approx(0.55 * 1.91 - 1)
    # Push: p_win=0.4, p_loss=0.4, p_push=0.2
    assert expected_net_return(0.4, 1.91, p_loss=0.4, p_push=0.2) == pytest.approx(0.4 * 0.91 - 0.4)
    won = settle_moneyline_full_game(True, 1, 1.91)
    assert won.net_return == pytest.approx(0.91)
    lost = settle_moneyline_full_game(True, 0, 1.91)
    assert lost.net_return == pytest.approx(-1)
    voided = settle_moneyline_full_game(True, 1, 1.91, void=True)
    assert voided.net_return == 0.0
    unavailable = settle_moneyline_full_game(True, None, 1.91)
    assert unavailable.net_return is None


def test_example_research_candidate_interval_rule():
    # Matches the prompt's worked example: 55% at 1.91 with interval 49-61 -> PASS.
    decision = evaluate_research_eligibility(0.55, 0.49, 0.61, 1.91, 10.0, [])
    assert decision.status is ResearchStatus.PASS
    assert decision.expected_return == pytest.approx(0.0505)
    assert decision.expected_return_low == pytest.approx(0.49 * 1.91 - 1)


def test_probabilities_bounded_and_devig_sums():
    games, odds = make_synthetic_league(n_games=40)
    feat = assemble_features(games)
    assert feat["elo_home_win_prob"].between(0, 1).all()
    quotes = select_quote_at_horizon(odds)
    assert (quotes["market_home_prob"] > 0).all()
    assert (quotes["market_home_prob"] < 1).all()
    # Late snapshot must not be selected.
    assert not quotes["bookmaker"].str.contains("late").any()


def test_missing_odds_disable_returns():
    games, odds = make_synthetic_league(n_games=30)
    games["market_home_prob"] = np.nan
    games["home_decimal"] = np.nan
    games["home_american"] = np.nan
    games["quote_age_minutes"] = np.nan
    games["quote_time"] = pd.NaT
    games["odds_quality_flags"] = "BLOCK_NO_HISTORICAL_ODDS"
    feat = assemble_features(games)
    feat["p_logit"] = 0.55
    feat["p_logit_low"] = 0.50
    feat["p_logit_high"] = 0.60
    forecasts = build_forecast_frame(
        feat,
        model_col="p_logit",
        low_col="p_logit_low",
        high_col="p_logit_high",
        model_version="test",
        origin=DatasetOrigin.SYNTHETIC,
    )
    assert forecasts["expected_return"].isna().all()
    assert (forecasts["research_status"] == ResearchStatus.BLOCKED.value).all()
    summary = ledger_summary(settle_ledger(decisions_from_forecasts(forecasts)))
    assert summary["sum_net_return"] is None


def test_simulations_are_not_observed_games():
    power = two_sample_paired_mean_power(
        n_unique_games=400,
        effect=-0.01,
        sd_of_paired_difference=0.05,
        simulations=500,
        question="unit",
    )
    assert power.simulations == 500
    assert power.n_unique_games == 400
    se = standard_error_iid_mean(np.arange(10.0))
    moe = margin_of_error(se, 1.96)
    assert moe > 0


def test_failed_challenger_not_promoted_and_rollback(tmp_paths):
    registry = ModelRegistry(tmp_paths)
    champ = ModelVersion(
        model_id="champ-1",
        family="elo",
        created_at="2026-10-02T00:00:00Z",
        code_version="0.1.0",
        data_snapshot="real",
        feature_names=["elo_diff"],
        seed=1,
        notes="",
        metrics={"n_unique_games": 500, "log_loss": 0.65, "origin": "REAL_DERIVED"},
        origin="REAL_DERIVED",
    )
    registry.record(champ)
    registry.set_champion(champ)
    bad = ModelVersion(
        model_id="challenger-bad",
        family="logit",
        created_at="2026-10-02T00:00:00Z",
        code_version="0.1.0",
        data_snapshot="real",
        feature_names=["elo_diff"],
        seed=1,
        notes="",
        metrics={"n_unique_games": 500, "log_loss": 0.80, "origin": "REAL_DERIVED"},
        origin="REAL_DERIVED",
    )
    decision = evaluate_challenger(registry, bad, min_unique_games=200)
    assert decision.promoted is False
    assert registry.champion()["model_id"] == "champ-1"
    worse_still = ModelVersion(
        model_id="temp",
        family="logit",
        created_at="t",
        code_version="0",
        data_snapshot="x",
        feature_names=[],
        seed=0,
        notes="",
        metrics={"n_unique_games": 500, "log_loss": 0.66, "origin": "REAL_DERIVED"},
        origin="REAL_DERIVED",
    )
    registry.set_champion(worse_still)
    restored = registry.rollback("champ-1")
    assert restored["model_id"] == "champ-1"
    assert "not a promotion test" in reject_automatic_win_streak(7, 8)


def test_forecasts_immutable_after_results(tmp_paths):
    games, odds = make_synthetic_league(n_games=16)
    quotes = select_quote_at_horizon(odds)
    games = games.merge(quotes, how="left", left_on="game_id", right_on="odds_game_id", suffixes=("", "_q"))
    feat = assemble_features(games)
    feat["p_logit"] = 0.52
    feat["p_logit_low"] = 0.40
    feat["p_logit_high"] = 0.64
    first = build_forecast_frame(
        feat,
        model_col="p_logit",
        low_col="p_logit_low",
        high_col="p_logit_high",
        model_version="v1",
        origin=DatasetOrigin.SYNTHETIC,
    )
    mutated = first.copy()
    mutated["probability"] = 0.99
    frozen = freeze_forecasts(first, mutated)
    assert (frozen["probability"] == first["probability"]).all()


def test_synthetic_labeled_and_not_real_performance():
    games, _ = make_synthetic_league(n_games=10)
    assert (games["origin"] == "SYNTHETIC").all()
    assert games["quality_flags"].str.contains("SYNTHETIC").all()
    feat = assemble_features(games)
    result, featured = walk_forward(
        feat,
        train_seasons=[2024],
        val_seasons=[2024],
        test_seasons=[2025],
        origin="SYNTHETIC",
    )
    assert result.origin == "SYNTHETIC"
    assert any("SYNTHETIC" in n for n in result.notes)


def test_end_to_end_synthetic_pipeline(tmp_paths, monkeypatch):
    from nhl_research import pipeline

    monkeypatch.setattr(pipeline, "Paths", lambda from_config=True: tmp_paths)
    payload = pipeline.run_synthetic_demo(tmp_paths)
    assert payload["origin"] == "SYNTHETIC"
    assert payload["metrics"]["logit"]["log_loss"] is not None
    assert payload["promotion"]["promoted"] is False
    html = (tmp_paths.reports / "index.html").read_text()
    assert "SYNTHETIC" in html
    forecasts = pd.read_json(tmp_paths.reports / "synthetic_forecasts.json")
    required = {
        "game_id",
        "prediction_time",
        "market_contract",
        "selection",
        "model_version",
        "data_as_of",
        "probability",
        "interval_method",
        "research_status",
        "reason",
    }
    assert required.issubset(forecasts.columns)

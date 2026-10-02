"""Chronological walk-forward evaluation. Random splits are not used."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from nhl_research.evaluation.metrics import MetricSet, probability_metrics, reliability_table
from nhl_research.features.assemble import CORE_MODEL_FEATURES, assemble_features
from nhl_research.models.baselines import elo_baseline, expanding_home_base_rate, market_baseline
from nhl_research.models.logistic import fit_logistic, predict_logistic
from nhl_research.uncertainty.power import two_sample_paired_mean_power
from nhl_research.uncertainty.resampling import paired_block_bootstrap


@dataclass
class WalkForwardResult:
    origin: str
    n_games: int
    metrics: dict[str, dict]
    paired_vs_market: dict | None
    power: dict | None
    reliability: dict[str, list]
    notes: list[str]


def walk_forward(
    games: pd.DataFrame,
    *,
    train_seasons: list[int],
    val_seasons: list[int],
    test_seasons: list[int],
    origin: str,
    seed: int = 20261002,
) -> tuple[WalkForwardResult, pd.DataFrame]:
    featured = assemble_features(games)
    train = featured[featured["season"].isin(train_seasons)].copy()
    val = featured[featured["season"].isin(val_seasons)].copy()
    test = featured[featured["season"].isin(test_seasons)].copy()
    notes = [
        "Chronological season split. Random train/test splits are not used.",
        "Calibration, if any, is fit on the validation seasons only.",
        "Games with null home_win are excluded from scoring but retained in coverage counts.",
    ]
    if origin == "SYNTHETIC":
        notes.append("SYNTHETIC_OFFLINE_FIXTURE: metrics are software demonstrations, not NHL results.")

    base = expanding_home_base_rate(featured)
    featured["p_base"] = base.probability
    featured["p_base_low"] = base.interval_low
    featured["p_base_high"] = base.interval_high

    elo = elo_baseline(featured)
    featured["p_elo"] = elo.probability
    featured["p_elo_low"] = elo.interval_low
    featured["p_elo_high"] = elo.interval_high

    mkt = market_baseline(featured)
    featured["p_market"] = mkt.probability

    artifact = fit_logistic(
        train,
        CORE_MODEL_FEATURES,
        val=val if not set(val_seasons) & set(train_seasons) else None,
        seed=seed,
        calibrate=not bool(set(val_seasons) & set(train_seasons)),
    )
    p, lo, hi = predict_logistic(artifact, featured)
    featured["p_logit"] = p
    featured["p_logit_low"] = lo
    featured["p_logit_high"] = hi
    featured["model_version"] = artifact.name

    test_f = featured[featured["season"].isin(test_seasons)].copy()
    y = pd.to_numeric(test_f["home_win"], errors="coerce").to_numpy()
    gids = test_f["game_id"].to_numpy()

    metrics = {
        "base_rate": asdict(probability_metrics(y, test_f["p_base"].to_numpy(), game_ids=gids, origin=origin)),
        "elo": asdict(probability_metrics(y, test_f["p_elo"].to_numpy(), game_ids=gids, origin=origin)),
        "logit": asdict(probability_metrics(y, test_f["p_logit"].to_numpy(), game_ids=gids, origin=origin)),
        "market": asdict(probability_metrics(y, test_f["p_market"].to_numpy(), game_ids=gids, origin=origin, notes="null where odds missing")),
    }

    paired = None
    power = None
    both = test_f.dropna(subset=["home_win", "p_logit", "p_market"]).copy()
    both["game_date"] = both.get("game_date", both["scheduled_start_utc"].astype(str).str[:10])
    if len(both) >= 20:
        both["ll_model"] = _row_logloss(both["home_win"], both["p_logit"])
        both["ll_market"] = _row_logloss(both["home_win"], both["p_market"])
        interval = paired_block_bootstrap(
            both,
            "ll_model",
            "ll_market",
            group_col="game_id",
            block_col="game_date",
            seed=seed,
            estimand="mean(logloss_logit - logloss_market)",
        )
        paired = interval.__dict__
        sd = float((both["ll_model"] - both["ll_market"]).std(ddof=1))
        if sd > 0:
            power = two_sample_paired_mean_power(
                n_unique_games=int(both["game_id"].nunique()),
                effect=-0.01,
                sd_of_paired_difference=sd,
                seed=seed,
                question="Detect a 0.01 improvement in mean log loss versus the same-time market",
            ).__dict__
    else:
        notes.append("Paired market comparison unavailable: fewer than 20 games with quotes and labels.")

    rel = {
        "logit": reliability_table(y, test_f["p_logit"].to_numpy()).to_dict(orient="records"),
        "elo": reliability_table(y, test_f["p_elo"].to_numpy()).to_dict(orient="records"),
    }
    result = WalkForwardResult(
        origin=origin,
        n_games=int(len(test_f)),
        metrics=metrics,
        paired_vs_market=paired,
        power=power,
        reliability=rel,
        notes=notes,
    )
    return result, featured


def _row_logloss(y, p) -> pd.Series:
    yv = pd.to_numeric(y, errors="coerce")
    pv = np.clip(pd.to_numeric(p, errors="coerce"), 1e-6, 1 - 1e-6)
    return -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))

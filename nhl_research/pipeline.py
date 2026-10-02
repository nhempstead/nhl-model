"""End-to-end research run used by the CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nhl_research.config import Paths, load_config
from nhl_research.dashboard.render import render_dashboard
from nhl_research.data.ingest import attach_quotes, ingest_featured_matchups, ingest_odds_local, ingest_synthetic
from nhl_research.data.inventory import inventory_payload
from nhl_research.data.odds import select_quote_at_horizon
from nhl_research.data.store import Warehouse, write_json
from nhl_research.entities import DatasetOrigin
from nhl_research.evaluation.replay import build_forecast_frame, freeze_forecasts
from nhl_research.evaluation.walkforward import walk_forward
from nhl_research.features.registry import registry_records
from nhl_research.ledger.paper import decisions_from_forecasts, ledger_summary, settle_ledger
from nhl_research.learning.lifecycle import evaluate_challenger
from nhl_research.models.registry import ModelRegistry, ModelVersion

UTC = timezone.utc


def run_synthetic_demo(paths: Paths | None = None) -> dict:
    paths = paths or Paths.from_config()
    warehouse = Warehouse(paths)
    ingest_synthetic(warehouse, n_games=80, seed=20261002)
    snapshots = warehouse.read_table("odds_snapshots")
    quotes = select_quote_at_horizon(snapshots, horizon_minutes=60)
    warehouse.replace_table(
        "quotes_tminus",
        quotes,
        origin=DatasetOrigin.SYNTHETIC,
        source="synthetic",
        note="T-60 quotes from synthetic snapshots; post-start rows dropped.",
        primary_key=["odds_game_id"],
    )
    attach_quotes(warehouse)
    games = warehouse.read_table("games_with_quotes")
    result, featured = walk_forward(
        games,
        train_seasons=[2024],
        val_seasons=[2024],
        test_seasons=[2025],
        origin=DatasetOrigin.SYNTHETIC.value,
    )
    forecasts = build_forecast_frame(
        featured,
        model_col="p_logit",
        low_col="p_logit_low",
        high_col="p_logit_high",
        model_version="logistic_data_only",
        origin=DatasetOrigin.SYNTHETIC,
    )
    existing = warehouse.read_table("forecasts")
    frozen = freeze_forecasts(existing, forecasts)
    warehouse.replace_table(
        "forecasts",
        frozen,
        origin=DatasetOrigin.SYNTHETIC,
        source="walk_forward",
        note="Immutable forecast records. SYNTHETIC_OFFLINE_FIXTURE.",
        primary_key=["game_id", "prediction_time", "model_version"],
    )
    ledger = settle_ledger(decisions_from_forecasts(frozen))
    warehouse.replace_table(
        "paper_ledger",
        ledger,
        origin=DatasetOrigin.SYNTHETIC,
        source="paper_ledger",
        note="Hypothetical units. SYNTHETIC_OFFLINE_FIXTURE.",
        primary_key=["game_id", "prediction_time", "model_version"],
    )
    summary = ledger_summary(ledger)
    registry = ModelRegistry(paths)
    challenger = ModelVersion(
        model_id="logit-synthetic-demo",
        family="logistic",
        created_at=datetime.now(tz=UTC).isoformat(),
        code_version="0.1.0",
        data_snapshot="synthetic",
        feature_names=["elo_diff", "rest_diff", "h_prior_games_min", "xg_pct_L20_diff"],
        seed=20261002,
        notes="Synthetic demonstration challenger; must not become a real-data champion.",
        metrics={
            "n_unique_games": result.metrics["logit"]["n_unique_games"],
            "log_loss": result.metrics["logit"]["log_loss"],
            "origin": DatasetOrigin.SYNTHETIC.value,
        },
        origin=DatasetOrigin.SYNTHETIC.value,
    )
    promotion = evaluate_challenger(registry, challenger, min_unique_games=200)
    payload = {
        "origin": DatasetOrigin.SYNTHETIC.value,
        "contract": {
            "target_season": "2026-2027",
            "primary_market": "moneyline_full_game",
            "horizon_minutes": 60,
            "hierarchy": [
                "engineering_validity",
                "data_integrity",
                "forecasting_quality",
                "uncertainty_assessment",
                "paper_trading_evidence",
            ],
        },
        "metrics": result.metrics,
        "paired_vs_market": result.paired_vs_market,
        "power": result.power,
        "ledger": summary,
        "promotion": promotion.__dict__,
        "forecasts_sample": forecasts.head(12).to_dict(orient="records"),
        "notes": result.notes,
        "feature_registry": registry_records(),
        "inventory": inventory_payload(),
    }
    write_json(paths.reports / "synthetic_run.json", payload)
    forecasts.to_json(paths.reports / "synthetic_forecasts.json", orient="records", date_format="iso")
    render_dashboard(paths, payload)
    featured.to_parquet(paths.warehouse / "featured_synthetic.parquet", index=False)
    return payload


def run_real_smoke(paths: Paths | None = None, max_games: int | None = None) -> dict:
    """Chronological evaluation when local matchup and odds files exist. Not a profit claim."""
    paths = paths or Paths.from_config()
    warehouse = Warehouse(paths)
    featured_csv = paths.processed / "matchups_featured.csv"
    odds_csv = paths.root / "data" / "historical_odds_api" / "nhl_odds_all.csv"
    if not featured_csv.exists() or featured_csv.stat().st_size < 1000:
        return {"status": "BLOCKED", "reason": "matchups_featured.csv missing or still an LFS pointer"}
    ingest_featured_matchups(warehouse, featured_csv)
    if odds_csv.exists() and odds_csv.stat().st_size > 1000:
        ingest_odds_local(warehouse, odds_csv, horizon_minutes=60)
        attach_quotes(warehouse)
        games = warehouse.read_table("games_with_quotes")
    else:
        games = warehouse.read_table("games")
        games["odds_quality_flags"] = "BLOCK_NO_HISTORICAL_ODDS"
    if max_games:
        games = games.sort_values(["season", "game_date", "game_id"]).tail(max_games)
    seasons = sorted(int(s) for s in games["season"].dropna().unique())
    if len(seasons) < 3:
        return {"status": "BLOCKED", "reason": "not enough seasons in featured file"}
    train = seasons[:-2]
    val = [seasons[-2]]
    test = [seasons[-1]]
    # Hold out the last season for a no-leakage test. If that season has no T-60 quotes,
    # also score the prior season so market comparison is not silently skipped.
    result, featured = walk_forward(
        games,
        train_seasons=train,
        val_seasons=val,
        test_seasons=test,
        origin=DatasetOrigin.REAL_DERIVED.value,
    )
    market_window = None
    if result.paired_vs_market is None and len(seasons) >= 4:
        mw_train = seasons[:-3]
        mw_val = [seasons[-3]]
        mw_test = [seasons[-2]]
        mw_result, mw_featured = walk_forward(
            games,
            train_seasons=mw_train,
            val_seasons=mw_val,
            test_seasons=mw_test,
            origin=DatasetOrigin.REAL_DERIVED.value,
        )
        market_window = {
            "split": {"train": mw_train, "val": mw_val, "test": mw_test},
            "metrics": mw_result.metrics,
            "paired_vs_market": mw_result.paired_vs_market,
            "power": mw_result.power,
            "notes": mw_result.notes
            + [
                "Secondary window used only because the newest test season has no T-60 quotes.",
                "This is still a single pre-specified fallback, not a search over seasons.",
            ],
        }
        mw_forecasts = build_forecast_frame(
            mw_featured[mw_featured["season"].isin(mw_test)],
            model_col="p_logit",
            low_col="p_logit_low",
            high_col="p_logit_high",
            model_version="logistic_data_only",
            origin=DatasetOrigin.REAL_DERIVED,
        )
        mw_forecasts.to_json(paths.reports / "real_smoke_forecasts_market_window.json", orient="records", date_format="iso")
        market_window["n_forecasts"] = int(len(mw_forecasts))
        market_window["ledger"] = ledger_summary(settle_ledger(decisions_from_forecasts(mw_forecasts)))
    forecasts = build_forecast_frame(
        featured[featured["season"].isin(test)],
        model_col="p_logit",
        low_col="p_logit_low",
        high_col="p_logit_high",
        model_version="logistic_data_only",
        origin=DatasetOrigin.REAL_DERIVED,
    )
    warehouse.replace_table(
        "forecasts_real_smoke",
        forecasts,
        origin=DatasetOrigin.REAL_DERIVED,
        source="matchups_featured+odds",
        note="Real derived historical evaluation. Paper returns null where T-60 quotes missing.",
        primary_key=["game_id", "prediction_time", "model_version"],
    )
    ledger = settle_ledger(decisions_from_forecasts(forecasts))
    summary = ledger_summary(ledger)
    payload = {
        "origin": DatasetOrigin.REAL_DERIVED.value,
        "split": {"train": train, "val": val, "test": test},
        "metrics": result.metrics,
        "paired_vs_market": result.paired_vs_market,
        "power": result.power,
        "ledger": summary,
        "market_overlap_window": market_window,
        "n_forecasts": int(len(forecasts)),
        "notes": result.notes
        + [
            "This smoke uses precomputed rolling features from matchups_featured.csv.",
            "Tied-goal games have null labels and are excluded from scoring.",
            "Local odds are sparse and mostly Bovada; this is not a Pinnacle closing-line study.",
        ],
        "forecasts_sample": forecasts.head(15).to_dict(orient="records"),
    }
    write_json(paths.reports / "real_smoke.json", payload)
    forecasts.to_json(paths.reports / "real_smoke_forecasts.json", orient="records", date_format="iso")
    render_dashboard(paths, payload)
    return payload

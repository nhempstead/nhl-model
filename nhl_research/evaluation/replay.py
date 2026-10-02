"""Build immutable forecast records and attach research eligibility."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nhl_research.entities import DatasetOrigin, ForecastRecord, ResearchStatus
from nhl_research.markets.conversion import american_to_decimal, expected_net_return
from nhl_research.markets.eligibility import evaluate_research_eligibility
from nhl_research.timeutil import forecast_time, to_iso

UTC = timezone.utc


def build_forecast_frame(
    featured: pd.DataFrame,
    *,
    model_col: str,
    low_col: str,
    high_col: str,
    model_version: str,
    horizon_minutes: int = 60,
    origin: DatasetOrigin,
) -> pd.DataFrame:
    rows = []
    for rec in featured.itertuples(index=False):
        start = rec.scheduled_start_utc
        start_dt = pd.Timestamp(start).to_pydatetime()
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        pred_time = forecast_time(start_dt, horizon_minutes)
        p = _f(getattr(rec, model_col, None))
        lo = _f(getattr(rec, low_col, None))
        hi = _f(getattr(rec, high_col, None))
        mkt = _f(getattr(rec, "market_home_prob", None))
        american = getattr(rec, "home_american", None)
        decimal = _f(getattr(rec, "home_decimal", None))
        if decimal is None and american is not None and pd.notna(american):
            try:
                decimal = american_to_decimal(int(american))
            except Exception:
                decimal = None
        quote_time = getattr(rec, "quote_time", None)
        age = _f(getattr(rec, "quote_age_minutes", None))
        flags = str(getattr(rec, "quality_flags", "") or "").split("|")
        flags += str(getattr(rec, "odds_quality_flags", "") or "").split("|")
        flags = [f for f in flags if f]
        origin_syn = origin == DatasetOrigin.SYNTHETIC or origin.value == "SYNTHETIC"
        decision = evaluate_research_eligibility(
            p,
            lo,
            hi,
            decimal,
            age,
            flags,
            origin_synthetic=origin_syn,
        )
        record = ForecastRecord(
            game_id=str(rec.game_id),
            prediction_time=pred_time,
            market_contract="moneyline_full_game|home|full_game_incl_ot_so",
            selection="home",
            model_version=model_version,
            data_as_of=pred_time,
            probability=p,
            interval_method="model_specific",
            interval_low=lo,
            interval_high=hi,
            interval_level=0.95,
            market_probability=mkt,
            quote_american=int(american) if american is not None and pd.notna(american) else None,
            quote_decimal=decimal,
            quote_time=_ts(quote_time),
            expected_return=decision.expected_return,
            expected_return_low=decision.expected_return_low,
            expected_return_high=decision.expected_return_high,
            data_quality_flags=flags,
            research_status=decision.status,
            reason=decision.reason,
            origin=origin,
            settled_outcome=int(rec.home_win) if rec.home_win is not None and pd.notna(rec.home_win) else None,
        )
        payload = record.dump()
        payload["home_team"] = rec.home_team
        payload["away_team"] = rec.away_team
        payload["season"] = int(rec.season)
        payload["game_date"] = str(getattr(rec, "game_date", ""))
        rows.append(payload)
    return pd.DataFrame(rows)


def freeze_forecasts(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Original forecasts remain immutable; duplicates keep the first record."""
    if existing is None or existing.empty:
        return new.drop_duplicates(subset=["game_id", "prediction_time", "model_version"], keep="first")
    combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset=["game_id", "prediction_time", "model_version"], keep="first")


def _f(value) -> float | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ts(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()

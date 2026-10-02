"""Hypothesis-driven feature registry. Inclusion is explicit, not automatic."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    hypothesis: str
    formula: str
    units: str
    denominator: str
    exposure: str
    lag: str
    source: str
    availability: str
    missing_treatment: str
    leakage_risks: str
    included_in_core: bool
    status: str


REGISTRY: list[FeatureSpec] = [
    FeatureSpec(
        name="elo_diff",
        family="team_strength",
        hypothesis="Opponent-adjusted cumulative results carry team quality between games.",
        formula="home_pregame_elo - away_pregame_elo, logistic map p = 1/(1+10^(-diff/400))",
        units="Elo points",
        denominator="games previously played by that team",
        exposure="one update per settled game",
        lag="shift after settlement; never the current game",
        source="canonical games",
        availability="after prior game available_at",
        missing_treatment="league mean Elo (1500); flag LOW_EXPOSURE if <10 prior games",
        leakage_risks="Using current-game margin or post-game ratings.",
        included_in_core=True,
        status="IMPLEMENTED",
    ),
    FeatureSpec(
        name="rest_days_diff",
        family="schedule_travel",
        hypothesis="Additional rest is associated with higher full-game win probability.",
        formula="days between a team's previous scheduled_start and this start, home minus away",
        units="days",
        denominator="calendar time between starts",
        exposure="requires a prior game",
        lag="previous game only",
        source="canonical schedule",
        availability="prior scheduled_start known at forecast time",
        missing_treatment="null + flag MISSING_REST; not filled with 0",
        leakage_risks="Using a restated start time that was unknown at the forecast timestamp.",
        included_in_core=True,
        status="IMPLEMENTED",
    ),
    FeatureSpec(
        name="home_indicator_absorbed",
        family="team_strength",
        hypothesis="Home teams win more often; modeled as intercept / Elo home offset.",
        formula="Elo home offset default 50 points (~54% vs equal opponent) estimated in-sample only on training folds",
        units="probability / Elo",
        denominator="settled full-game results",
        exposure="league-wide",
        lag="training window only",
        source="canonical games",
        availability="historical settled games",
        missing_treatment="n/a",
        leakage_risks="Estimating home advantage on the evaluation season.",
        included_in_core=True,
        status="IMPLEMENTED",
    ),
    FeatureSpec(
        name="xg_pct_L20_diff",
        family="shot_generation_quality",
        hypothesis="Recent 5-on-5 expected-goal share is a more stable quality signal than goal share.",
        formula="mean of prior 20 games of xGoalsFor/(xGoalsFor+xGoalsAgainst), home minus away",
        units="proportion",
        denominator="5-on-5 xG for + against",
        exposure="min 10 prior 5-on-5 games",
        lag="shift(1) rolling",
        source="MoneyPuck 5on5 game rows or featured precompute",
        availability="after prior game publication lag",
        missing_treatment="null, not zero; model may drop or flag",
        leakage_risks="MoneyPuck xG is a model; historical version unknown. Season-to-date files include later games and are rejected as pregame inputs.",
        included_in_core=True,
        status="IMPLEMENTED",
    ),
    FeatureSpec(
        name="market_home_prob_tminus",
        family="market_information",
        hypothesis="Same-time de-vigged moneyline is a strong probability baseline.",
        formula="proportional de-vig of the same bookmaker's two-way quote at or before T-horizon",
        units="probability",
        denominator="home+away implied odds at one timestamp",
        exposure="requires complete two-way market",
        lag="snapshot_time <= prediction_time",
        source="odds snapshots",
        availability="snapshot_time",
        missing_treatment="null; paper returns blocked rather than imputed",
        leakage_risks="Closing line, best-price stitching, or post-start snapshots.",
        included_in_core=True,
        status="IMPLEMENTED",
    ),
    FeatureSpec(
        name="realized_starting_goalie",
        family="goaltending",
        hypothesis="Starter quality changes win probability, but only if known before the cutoff.",
        formula="not in core; post-game participation is not a pregame confirmation",
        units="n/a",
        denominator="n/a",
        exposure="n/a",
        lag="unknown pregame",
        source="shots-derived goalie_starts",
        availability="after the game",
        missing_treatment="unknown remains unknown",
        leakage_risks="Inferring the starter from who played.",
        included_in_core=False,
        status="NOT_YET_IMPLEMENTED",
    ),
]


def core_feature_names() -> list[str]:
    return [spec.name for spec in REGISTRY if spec.included_in_core]


def registry_records() -> list[dict]:
    return [spec.__dict__ for spec in REGISTRY]

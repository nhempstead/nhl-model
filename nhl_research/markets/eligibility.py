"""Research eligibility rules. PASS means criteria were not satisfied."""

from __future__ import annotations

from dataclasses import dataclass

from nhl_research.entities import ResearchStatus
from nhl_research.markets.conversion import expected_net_return


@dataclass(frozen=True)
class EligibilityDecision:
    status: ResearchStatus
    reason: str
    expected_return: float | None
    expected_return_low: float | None
    expected_return_high: float | None


def evaluate_research_eligibility(
    probability: float | None,
    interval_low: float | None,
    interval_high: float | None,
    decimal_odds: float | None,
    quote_age_minutes: float | None,
    flags: list[str],
    *,
    require_lower_bound_positive: bool = True,
    max_interval_width: float = 0.25,
    max_quote_age_minutes: float = 180.0,
    origin_synthetic: bool = False,
) -> EligibilityDecision:
    if origin_synthetic:
        flags = list(flags) + ["SYNTHETIC_OFFLINE_FIXTURE"]
    blocking = [f for f in flags if f.startswith("BLOCK_")]
    if probability is None:
        return EligibilityDecision(ResearchStatus.BLOCKED, "model probability unavailable", None, None, None)
    if decimal_odds is None:
        return EligibilityDecision(
            ResearchStatus.BLOCKED,
            "historical or same-time quote unavailable; return not reported",
            None,
            None,
            None,
        )
    if blocking:
        return EligibilityDecision(
            ResearchStatus.BLOCKED,
            "; ".join(blocking),
            None,
            None,
            None,
        )
    point = expected_net_return(probability, decimal_odds)
    low = expected_net_return(interval_low, decimal_odds) if interval_low is not None else None
    high = expected_net_return(interval_high, decimal_odds) if interval_high is not None else None
    reasons: list[str] = []
    if interval_low is None or interval_high is None:
        reasons.append("probability interval unavailable")
        return EligibilityDecision(ResearchStatus.BLOCKED, "; ".join(reasons), point, None, None)
    if (interval_high - interval_low) > max_interval_width:
        reasons.append("probability interval wider than configured maximum")
    if quote_age_minutes is not None and quote_age_minutes > max_quote_age_minutes:
        reasons.append("quote older than freshness limit")
    if require_lower_bound_positive and low is not None and low <= 0:
        reasons.append("lower expected-return bound is not positive")
    if reasons:
        return EligibilityDecision(ResearchStatus.PASS, "; ".join(reasons), point, low, high)
    return EligibilityDecision(
        ResearchStatus.RESEARCH_CANDIDATE,
        "configured research criteria satisfied",
        point,
        low,
        high,
    )

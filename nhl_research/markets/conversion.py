"""American odds conversion, proportional de-vigging, and expected-return math."""

from __future__ import annotations

from dataclasses import dataclass

from nhl_research.exceptions import InvalidOddsError


def american_to_decimal(american: int | float) -> float:
    """Convert American odds A to decimal odds.

    If A > 0: 1 + A/100
    If A < 0: 1 + 100/abs(A)
    """
    try:
        value = float(american)
    except (TypeError, ValueError) as exc:
        raise InvalidOddsError(f"American odds not numeric: {american!r}") from exc
    if value != value:  # NaN
        raise InvalidOddsError("American odds is NaN")
    if value == 0:
        raise InvalidOddsError("American odds of 0 is invalid")
    if value > 0:
        decimal = 1.0 + value / 100.0
    else:
        decimal = 1.0 + 100.0 / abs(value)
    if decimal <= 1.0:
        raise InvalidOddsError(f"Implied decimal odds must exceed 1, got {decimal}")
    return decimal


def decimal_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise InvalidOddsError(f"Decimal odds must exceed 1, got {decimal_odds}")
    return 1.0 / decimal_odds


def american_to_implied_probability(american: int | float) -> float:
    return decimal_to_implied_probability(american_to_decimal(american))


@dataclass(frozen=True)
class DeVigResult:
    method: str
    raw_implied: tuple[float, ...]
    de_vigged: tuple[float, ...]
    overround: float
    note: str


def proportional_devig(decimal_odds: list[float] | tuple[float, ...]) -> DeVigResult:
    """Normalize implied probabilities from one complete same-book market.

    This is a de-vigging *convention*, not recovered ground truth.
    Unrelated best prices from different books or times must not be mixed.
    """
    if len(decimal_odds) < 2:
        raise InvalidOddsError("De-vigging requires a complete market of at least two outcomes")
    implied = tuple(decimal_to_implied_probability(x) for x in decimal_odds)
    total = sum(implied)
    if total <= 0:
        raise InvalidOddsError("Implied probabilities sum to zero")
    normalized = tuple(q / total for q in implied)
    if abs(sum(normalized) - 1.0) > 1e-12:
        raise InvalidOddsError("De-vigged probabilities failed to sum to 1")
    return DeVigResult(
        method="proportional_same_book_same_timestamp",
        raw_implied=implied,
        de_vigged=normalized,
        overround=total - 1.0,
        note=(
            "Proportional (multiplicative) de-vig of a complete mutually exclusive "
            "market from one bookmaker at one timestamp. Not a true-probability estimator."
        ),
    )


def expected_net_return(
    p_win: float,
    decimal_odds: float,
    p_loss: float | None = None,
    p_push: float = 0.0,
) -> float:
    """Hypothetical one-unit net return.

    Without a push: p_win * decimal_odds - 1
    With win/loss/push: p_win * (decimal_odds - 1) - p_loss
    Push returns the stake (net 0). Costs are not subtracted here.
    """
    _validate_probability(p_win, "p_win")
    if p_push < 0 or p_push > 1:
        raise InvalidOddsError("p_push must be in [0, 1]")
    if p_loss is None:
        p_loss = 1.0 - p_win - p_push
    _validate_probability(p_loss, "p_loss")
    total = p_win + p_loss + p_push
    if abs(total - 1.0) > 1e-8:
        raise InvalidOddsError(f"p_win + p_loss + p_push must equal 1, got {total}")
    if decimal_odds <= 1.0:
        raise InvalidOddsError("Decimal odds must exceed 1")
    return p_win * (decimal_odds - 1.0) - p_loss


def _validate_probability(value: float, name: str) -> None:
    if value < 0 or value > 1 or value != value:
        raise InvalidOddsError(f"{name} must be a probability in [0, 1], got {value}")

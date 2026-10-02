"""Versioned settlement rules for hypothetical paper positions."""

from __future__ import annotations

from dataclasses import dataclass

from nhl_research.entities import SettlementResult

SETTLEMENT_RULES_VERSION = "moneyline_full_game_v1"


@dataclass(frozen=True)
class SettledPosition:
    result: SettlementResult
    net_return: float | None
    rules_version: str
    reason: str


def settle_moneyline_full_game(
    selection_is_home: bool,
    home_win: int | None,
    decimal_odds: float,
    void: bool = False,
) -> SettledPosition:
    """Settle a full-game NHL moneyline. Regular-season moneylines have no push.

    A MoneyPuck goal tie is treated as UNAVAILABLE rather than a loss: shootouts
    are not reliably encoded as a goal in that source.
    """
    if void:
        return SettledPosition(SettlementResult.VOID, 0.0, SETTLEMENT_RULES_VERSION, "voided")
    if home_win is None:
        return SettledPosition(
            SettlementResult.UNAVAILABLE,
            None,
            SETTLEMENT_RULES_VERSION,
            "official moneyline winner unavailable",
        )
    won = (home_win == 1 and selection_is_home) or (home_win == 0 and not selection_is_home)
    if won:
        return SettledPosition(
            SettlementResult.WIN,
            decimal_odds - 1.0,
            SETTLEMENT_RULES_VERSION,
            "selection won including overtime/shootout if applicable",
        )
    return SettledPosition(
        SettlementResult.LOSS,
        -1.0,
        SETTLEMENT_RULES_VERSION,
        "selection lost",
    )

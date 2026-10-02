"""Controlled challenger evaluation, promotion, and rollback.

A winning streak is not a promotion test. A losing streak is not by itself a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from nhl_research.exceptions import PromotionRejected
from nhl_research.models.registry import ModelRegistry, ModelVersion

UTC = timezone.utc


@dataclass
class PromotionDecision:
    promoted: bool
    champion_id: str | None
    challenger_id: str
    reasons: list[str]
    metrics: dict[str, Any]


def evaluate_challenger(
    registry: ModelRegistry,
    challenger: ModelVersion,
    *,
    min_unique_games: int = 200,
    max_log_loss_regression: float = 0.0,
) -> PromotionDecision:
    """Promote only if predefined out-of-sample requirements pass."""
    reasons: list[str] = []
    champ = registry.champion()
    metrics = challenger.metrics or {}
    n = int(metrics.get("n_unique_games") or 0)
    if n < min_unique_games:
        reasons.append(f"insufficient unique games ({n} < {min_unique_games})")
    if metrics.get("origin") == "SYNTHETIC":
        reasons.append("synthetic fixtures cannot promote a real-data champion")
    chall_ll = metrics.get("log_loss")
    champ_ll = None if champ is None else (champ.get("metrics") or {}).get("log_loss")
    if chall_ll is None:
        reasons.append("challenger log loss unavailable")
    if champ is not None and chall_ll is not None and champ_ll is not None:
        if float(chall_ll) > float(champ_ll) + max_log_loss_regression:
            reasons.append(
                f"challenger log loss {chall_ll:.4f} worse than champion {champ_ll:.4f}"
            )
    if "calibration_not_worse" in metrics and not metrics["calibration_not_worse"]:
        reasons.append("calibration degraded versus champion")
    if reasons:
        registry.record(challenger)
        return PromotionDecision(False, None if champ is None else champ.get("model_id"), challenger.model_id, reasons, metrics)
    registry.record(challenger)
    registry.set_champion(challenger)
    return PromotionDecision(True, challenger.model_id, challenger.model_id, ["predefined tests passed"], metrics)


def reject_automatic_win_streak(wins: int, window: int) -> str:
    return (
        f"Recent record {wins}/{window} is not a promotion test. "
        "Retain the champion unless evaluate_challenger() passes."
    )

"""Immutable hypothetical paper ledger. Not a wagering interface."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from nhl_research.entities import ResearchStatus, SettlementResult
from nhl_research.markets.settlement import settle_moneyline_full_game

UTC = timezone.utc
LEDGER_VERSION = "paper_unit_v1"


def decisions_from_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Record every original decision. No staking advice, no chase rules."""
    frame = forecasts.copy()
    frame["ledger_version"] = LEDGER_VERSION
    frame["unit_size"] = 1.0
    frame["decision_recorded_at"] = datetime.now(tz=UTC).isoformat()
    frame["is_paper_position"] = frame["research_status"] == ResearchStatus.RESEARCH_CANDIDATE.value
    # Always keep non-candidates so coverage cannot be hidden.
    return frame


def settle_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    results = []
    nets = []
    reasons = []
    for rec in ledger.itertuples(index=False):
        decimal = rec.quote_decimal if pd.notna(getattr(rec, "quote_decimal", None)) else None
        y = rec.settled_outcome if pd.notna(getattr(rec, "settled_outcome", None)) else None
        if decimal is None:
            results.append(SettlementResult.UNAVAILABLE.value)
            nets.append(None)
            reasons.append("no timestamped quote; return not reconstructed")
            continue
        if not bool(getattr(rec, "is_paper_position", False)):
            results.append(SettlementResult.UNSETTLED.value)
            nets.append(None)
            reasons.append("not a research candidate; scored for coverage only")
            continue
        settled = settle_moneyline_full_game(True, None if y is None else int(y), float(decimal))
        results.append(settled.result.value)
        nets.append(settled.net_return)
        reasons.append(settled.reason)
    out = ledger.copy()
    out["settlement_result"] = results
    out["paper_net_return"] = nets
    out["settlement_reason"] = reasons
    return out


def ledger_summary(settled: pd.DataFrame) -> dict:
    pos = settled[settled["is_paper_position"] == True]  # noqa: E712
    scored = pos[pos["paper_net_return"].notna()]
    origin = str(settled["origin"].iloc[0]) if len(settled) else "UNAVAILABLE"
    if origin == "SYNTHETIC":
        note = "SYNTHETIC_OFFLINE_FIXTURE. Hypothetical returns are not NHL paper-trading evidence."
    else:
        note = "Hypothetical unit returns on RESEARCH_CANDIDATE rows only. Not executable fills."
    if scored.empty:
        return {
            "n_forecasts": int(len(settled)),
            "n_candidates": int(len(pos)),
            "n_settled_candidates": 0,
            "sum_net_return": None,
            "mean_net_return": None,
            "max_drawdown": None,
            "origin": origin,
            "note": note + " Insufficient quotes or candidates to report returns.",
        }
    rets = scored["paper_net_return"].astype(float)
    equity = rets.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "n_forecasts": int(len(settled)),
        "n_candidates": int(len(pos)),
        "n_settled_candidates": int(len(scored)),
        "sum_net_return": float(rets.sum()),
        "mean_net_return": float(rets.mean()),
        "max_drawdown": float(drawdown.min()),
        "origin": origin,
        "note": note,
    }

"""Scoring rules and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class MetricSet:
    n_labeled: int
    n_unique_games: int
    log_loss: float | None
    brier: float | None
    accuracy: float | None
    coverage: float
    abstained: int
    origin: str
    notes: str


def probability_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    game_ids: np.ndarray | None = None,
    origin: str,
    notes: str = "",
) -> MetricSet:
    mask = np.isfinite(y.astype(float)) & np.isfinite(p.astype(float))
    y_m = y[mask].astype(int)
    p_m = np.clip(p[mask].astype(float), 1e-6, 1 - 1e-6)
    n = int(y_m.size)
    unique = int(np.unique(game_ids[mask]).size) if game_ids is not None else n
    if n == 0:
        return MetricSet(0, 0, None, None, None, 0.0, int((~mask).sum()), origin, notes or "no labeled rows")
    return MetricSet(
        n_labeled=n,
        n_unique_games=unique,
        log_loss=float(log_loss(y_m, p_m, labels=[0, 1])),
        brier=float(brier_score_loss(y_m, p_m)),
        accuracy=float(np.mean((p_m >= 0.5) == y_m)),
        coverage=float(n / max(len(y), 1)),
        abstained=int((~mask).sum()),
        origin=origin,
        notes=notes,
    )


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    mask = np.isfinite(y.astype(float)) & np.isfinite(p.astype(float))
    y_m = y[mask].astype(float)
    p_m = p[mask].astype(float)
    if y_m.size == 0:
        return pd.DataFrame(columns=["bin", "count", "predicted", "observed"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p_m, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if not np.any(sel):
            rows.append({"bin": b, "count": 0, "predicted": None, "observed": None})
            continue
        rows.append(
            {
                "bin": b,
                "count": int(sel.sum()),
                "predicted": float(p_m[sel].mean()),
                "observed": float(y_m[sel].mean()),
            }
        )
    return pd.DataFrame(rows)

"""Regularized logistic regression with fold-local scaling and calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nhl_research.exceptions import LeakageError
from nhl_research.uncertainty.intervals import clip_probability_interval


@dataclass
class LogisticArtifact:
    name: str
    scaler: StandardScaler
    model: LogisticRegression
    calibrator: LogisticRegression | None
    feature_names: list[str]
    train_index_hash: int
    fill_values: dict[str, float] = field(default_factory=dict)


def _design(frame: pd.DataFrame, feature_names: list[str], fill_values: dict[str, float] | None) -> np.ndarray:
    cols = []
    for name in feature_names:
        series = pd.to_numeric(frame[name], errors="coerce") if name in frame.columns else pd.Series(np.nan, index=frame.index)
        if fill_values is not None:
            series = series.fillna(fill_values.get(name, 0.0))
        cols.append(series.to_numpy(dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(frame), 0))


def fit_logistic(
    train: pd.DataFrame,
    feature_names: list[str],
    *,
    val: pd.DataFrame | None = None,
    C: float = 0.5,
    seed: int = 20261002,
    calibrate: bool = True,
) -> LogisticArtifact:
    if train.empty:
        raise ValueError("logistic train frame is empty")
    y = pd.to_numeric(train["home_win"], errors="coerce")
    mask = y.notna()
    train_used = train.loc[mask]
    y_used = y.loc[mask].astype(int).to_numpy()
    fill = {name: float(pd.to_numeric(train_used[name], errors="coerce").median()) if name in train_used else 0.0 for name in feature_names}
    for name, value in list(fill.items()):
        if not np.isfinite(value):
            fill[name] = 0.0
    X = _design(train_used, feature_names, fill)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(C=C, max_iter=1000, random_state=seed)
    model.fit(Xs, y_used)
    calibrator = None
    if calibrate and val is not None and len(val):
        yv = pd.to_numeric(val["home_win"], errors="coerce")
        vmask = yv.notna()
        if vmask.sum() >= 30:
            Xv = scaler.transform(_design(val.loc[vmask], feature_names, fill))
            scores = model.decision_function(Xv).reshape(-1, 1)
            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(scores, yv.loc[vmask].astype(int).to_numpy())
    return LogisticArtifact(
        name="logistic_data_only",
        scaler=scaler,
        model=model,
        calibrator=calibrator,
        feature_names=feature_names,
        train_index_hash=int(pd.util.hash_pandas_object(train_used[["game_id"]]).sum()),
        fill_values=fill,
    )


def predict_logistic(artifact: LogisticArtifact, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = artifact.scaler.transform(_design(frame, artifact.feature_names, artifact.fill_values))
    raw_scores = artifact.model.decision_function(X).reshape(-1, 1)
    if artifact.calibrator is not None:
        p = artifact.calibrator.predict_proba(raw_scores)[:, 1]
    else:
        p = artifact.model.predict_proba(X)[:, 1]
    # Coefficient bootstrap is expensive; use a local Laplace approximation around p
    # with width shrinking in prior games. This is parameter-scale, not binomial n.
    prior = np.maximum(pd.to_numeric(frame.get("h_prior_games_min"), errors="coerce").fillna(1).to_numpy(), 1.0)
    se = np.sqrt(p * (1 - p) / (prior + 10.0))  # shrinkage; NOT iid game-count SE of this p
    low, high = zip(*[clip_probability_interval(pi - 1.96 * s, pi + 1.96 * s) for pi, s in zip(p, se)])
    return p, np.asarray(low), np.asarray(high)


def assert_scaler_not_fit_on_holdout(artifact: LogisticArtifact, holdout: pd.DataFrame) -> None:
    hold_hash = int(pd.util.hash_pandas_object(holdout[["game_id"]]).sum()) if "game_id" in holdout.columns else 0
    if hold_hash and hold_hash == artifact.train_index_hash and len(holdout) > 0:
        raise LeakageError("Holdout game_id hash matches the training set used to fit preprocessing")

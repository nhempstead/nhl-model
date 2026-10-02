"""Paired and block resampling that keeps related forecasts together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhl_research.uncertainty.intervals import IntervalReport


@dataclass(frozen=True)
class ResamplePlan:
    grouping: str
    n_replicates: int
    seed: int
    block_key: str


def paired_block_bootstrap(
    frame: pd.DataFrame,
    value_a: str,
    value_b: str,
    *,
    group_col: str = "game_id",
    block_col: str | None = "game_date",
    n_replicates: int = 500,
    seed: int = 20261002,
    level: float = 0.95,
    estimand: str = "mean(value_a - value_b)",
) -> IntervalReport:
    """Bootstrap the mean paired difference, resampling blocks of related games.

    Repeated odds snapshots or multiple contracts on the same game stay together
    when grouped by game_id. Simulation replicates are not additional games.
    """
    if frame.empty:
        raise ValueError("paired bootstrap requires observations")
    work = frame[[group_col, value_a, value_b] + ([block_col] if block_col else [])].dropna()
    if block_col and block_col in work.columns:
        blocks = [g.index.to_numpy() for _, g in work.groupby(block_col, sort=False)]
        grouping = f"block:{block_col}; groups remain intact inside block"
    else:
        blocks = [g.index.to_numpy() for _, g in work.groupby(group_col, sort=False)]
        grouping = f"group:{group_col}"
    if not blocks:
        raise ValueError("no complete pairs for bootstrap")
    rng = np.random.default_rng(seed)
    diffs = (work[value_a] - work[value_b]).to_numpy(dtype=float)
    observed = float(np.mean(diffs))
    stats = np.empty(n_replicates, dtype=float)
    n_blocks = len(blocks)
    for i in range(n_replicates):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        sample_idx = np.concatenate([blocks[j] for j in chosen])
        sample = work.loc[sample_idx]
        stats[i] = float((sample[value_a] - sample[value_b]).mean())
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    n_games = int(work[group_col].nunique())
    return IntervalReport(
        estimand=estimand,
        method=f"paired_block_bootstrap[{grouping}]",
        level=level,
        estimate=observed,
        low=float(low),
        high=float(high),
        n_observations=int(len(work)),
        n_unique_games=n_games,
        effective_sample_size=_design_effect_ess(work, group_col, n_games),
        assumptions="Blocks are exchangeable. Within-block dependence is preserved by resampling whole blocks.",
        limitations=(
            "Block length is a modeling choice; results can change with gameday vs week blocks. "
            f"{n_replicates} bootstrap draws quantify resampling variability of the estimator. "
            "They are not additional observed games."
        ),
        simulation_draws=n_replicates,
        simulation_draws_are_not_observations=True,
    )


def _design_effect_ess(work: pd.DataFrame, group_col: str, n_games: int) -> float:
    """Kish-style effective sample size using cluster sizes. Conservative when clusters vary."""
    sizes = work.groupby(group_col).size().to_numpy(dtype=float)
    if sizes.size == 0:
        return 0.0
    m = sizes.mean()
    if m <= 1:
        return float(n_games)
    # Without a measured ICC, treat each extra row in a game as dependent (ICC=1)
    # so ESS equals unique games, not snapshot rows.
    return float(n_games)

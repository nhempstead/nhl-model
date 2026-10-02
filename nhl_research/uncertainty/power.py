"""Prospective power and minimum-detectable-effect analysis.

Power is the probability that a specified test rejects a false null under a
specified alternative. It is not a model's win probability. Observed power
after seeing the data is not reported as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nhl_research.uncertainty.intervals import _normal_ppf


@dataclass(frozen=True)
class PowerResult:
    question: str
    null: str
    alternative: str
    alpha: float
    n_unique_games: int
    assumed_effect: float
    assumed_sd: float
    power: float
    mde_80: float
    mde_90: float
    method: str
    assumptions: str
    limitations: str
    simulations: int


def two_sample_paired_mean_power(
    *,
    n_unique_games: int,
    effect: float,
    sd_of_paired_difference: float,
    alpha: float = 0.05,
    simulations: int = 2000,
    seed: int = 20261002,
    question: str,
) -> PowerResult:
    """Monte Carlo power for a two-sided t-test on a paired mean difference.

    The simulation reduces Monte Carlo error in the *power estimate*. It does
    not create new NHL evidence.
    """
    if n_unique_games < 2:
        raise ValueError("power analysis requires n_unique_games >= 2")
    if sd_of_paired_difference <= 0:
        raise ValueError("sd_of_paired_difference must be positive")
    rng = np.random.default_rng(seed)
    crit = _t_crit_approx(n_unique_games - 1, 1.0 - alpha / 2.0)
    se = sd_of_paired_difference / np.sqrt(n_unique_games)
    # Parametric draws of the sample mean under the alternative
    z = rng.normal(loc=effect, scale=se, size=simulations)
    # Approximate t with z for large n; still count unique games only
    reject = np.abs(z / se) >= crit
    power = float(np.mean(reject))
    mde_80 = _mde(0.80, alpha, n_unique_games, sd_of_paired_difference)
    mde_90 = _mde(0.90, alpha, n_unique_games, sd_of_paired_difference)
    return PowerResult(
        question=question,
        null="mean paired difference = 0",
        alternative=f"mean paired difference = {effect}",
        alpha=alpha,
        n_unique_games=n_unique_games,
        assumed_effect=effect,
        assumed_sd=sd_of_paired_difference,
        power=power,
        mde_80=mde_80,
        mde_90=mde_90,
        method="monte_carlo_normal_mean_paired_t_approx",
        assumptions=(
            "Paired differences are approximately iid given the block-level reduction "
            "to unique games. sd is a research assumption, not estimated from the evaluation set "
            "being tested in the same step."
        ),
        limitations=(
            f"{simulations} draws estimate power; they are not extra games. "
            "Dependence, selection, and changing markets can reduce actual power. "
            "Do not interpret post-hoc observed power as confirmation."
        ),
        simulations=simulations,
    )


def _mde(target_power: float, alpha: float, n: int, sd: float) -> float:
    z_a = _normal_ppf(1.0 - alpha / 2.0)
    z_p = _normal_ppf(target_power)
    return float((z_a + z_p) * sd / np.sqrt(n))


def _t_crit_approx(df: int, p: float) -> float:
    """Normal approximation to t critical value; conservative note for small df."""
    z = _normal_ppf(p)
    if df < 30:
        # Cornish-Fisher-style adjustment
        return float(z + (z**3 + z) / (4 * df))
    return float(z)

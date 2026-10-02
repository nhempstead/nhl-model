"""Sampling uncertainty, interval construction, and labeled estimands.

Precision here means sampling variability of an estimator (interval width),
not guaranteed closeness to the truth. A narrow interval can still be wrong
if the model is misspecified. See NIST e-Handbook 1.3.5.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nhl_research.exceptions import NHLResearchError


@dataclass(frozen=True)
class IntervalReport:
    estimand: str
    method: str
    level: float
    estimate: float
    low: float
    high: float
    n_observations: int
    n_unique_games: int
    effective_sample_size: float | None
    assumptions: str
    limitations: str
    simulation_draws: int | None = None
    simulation_draws_are_not_observations: bool = True


def standard_error_iid_mean(values: np.ndarray) -> float:
    """s / sqrt(n) for an iid sample mean. Do not apply to dependent games."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        raise NHLResearchError("iid standard error requires at least two observations")
    return float(np.std(x, ddof=1) / np.sqrt(n))


def margin_of_error(standard_error: float, critical_value: float) -> float:
    """Half-width of a symmetric interval: critical_value * SE.

    Lowering the confidence level narrows the interval without adding information.
    """
    if standard_error < 0:
        raise NHLResearchError("standard error must be non-negative")
    if critical_value <= 0:
        raise NHLResearchError("critical value must be positive")
    return float(critical_value * standard_error)


def mean_interval_iid(
    values: np.ndarray,
    *,
    level: float = 0.95,
    estimand: str,
) -> IntervalReport:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        raise NHLResearchError("mean interval requires at least two observations")
    if not 0 < level < 1:
        raise NHLResearchError("confidence level must be in (0, 1)")
    estimate = float(np.mean(x))
    se = standard_error_iid_mean(x)
    # Normal critical value for 95% is ~1.96; use scipy-free approximation via numpy
    crit = float(_normal_ppf(0.5 + level / 2.0))
    moe = margin_of_error(se, crit)
    return IntervalReport(
        estimand=estimand,
        method="iid_normal_mean",
        level=level,
        estimate=estimate,
        low=estimate - moe,
        high=estimate + moe,
        n_observations=n,
        n_unique_games=n,
        effective_sample_size=float(n),
        assumptions="Independent identically distributed observations; approximately normal sampling distribution of the mean.",
        limitations=(
            "Invalid for dependent games, repeated odds snapshots, or an individual "
            "game probability. Interval width is not a guarantee of accuracy."
        ),
    )


def _normal_ppf(p: float) -> float:
    """Acklam/rational approximation for the standard normal quantile."""
    if p <= 0.0 or p >= 1.0:
        raise NHLResearchError("normal quantile p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577509590705e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068071618818e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def clip_probability_interval(low: float, high: float) -> tuple[float, float]:
    return (float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0)))

# Statistical methods implemented in code

These are calculations and safeguards, not decorative labels.

## Precision

Implemented as interval width of a named estimand (`nhl_research.uncertainty.intervals`). Precision is sampling variability. A narrow interval does not prove the model is close to the truth (NIST e-Handbook 1.3.5.2).

## Standard error

`standard_error_iid_mean` = s / sqrt(n) **only** for an iid sample mean. It is not applied to an individual game probability merely because historical games exist. Aggregate metric uncertainty uses paired block bootstrap on unique games / gamedays (`resampling.py`). Repeated snapshots do not increase effective sample size; ESS equals unique games.

## Margin of error

`margin_of_error = critical_value * standard_error`. The software does not lower the confidence level to advertise a tighter interval. Default level is 0.95 from config.

## Probability vs outcome vs performance uncertainty

- **Probability uncertainty:** Elo rating-SE mapped through the logistic; logistic local interval using a shrunk variance (not binomial n of the league). Labeled in `interval_method`.
- **Outcome uncertainty:** Bernoulli(p) remaining even if p were known.
- **Performance uncertainty:** paired block bootstrap of mean log-loss difference; power analysis uses unique games only.

## Power

`two_sample_paired_mean_power` estimates the probability of rejecting a false null of zero mean paired log-loss difference under a specified alternative (default illustrative effect −0.01). Targets 80% and 90% MDEs are reported. Simulation draws are stored separately and **must not** be added to n_unique_games. Observed (post-hoc) power is not reported as evidence (NIST PRC 2.2.2).

## Multiple comparisons

Primary confirmatory contrast is mean log loss vs the same-time market on unique settled games. Other models, thresholds, and feature families are exploratory unless pre-registered in config.

## Sources

- NIST e-Handbook 1.3.5.2, confidence intervals: https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
- NIST e-Handbook 2.2.2, power: https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm
- Scikit-learn time-series split / leakage: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Scikit-learn calibration: https://scikit-learn.org/stable/modules/calibration.html

# Project contract — NHL 2026–27 forecasting research platform

**Status:** clean repository containing the implemented core (see docs/REQUIREMENT_MATRIX.md). The previous project’s scripts and data dumps are not included.  
**Execution date:** 2026-10-02 (UTC).  
**Target season:** 2026–2027 NHL regular season (started 2026-09-29; scheduled through 2027-04-10; 84 games per team).

## Objective

Estimate calibrated full-game moneyline probabilities at a documented forecast time. Market comparison and hypothetical paper-trading are downstream of those probabilities. Outperforming bookmakers is a **hypothesis to test**, not a design premise.

## Prediction target (phase 1)

- Market: full-game moneyline, including overtime and shootout.
- Selection: home win (away is the complement).
- Regulation-only, totals, puck lines, playoffs, player props, and in-play: **not in this phase**.

## Forecast timestamp

Primary horizon: **60 minutes before the scheduled start known at that time**.  
Do not replace a historical start time with a later revision. MoneyPuck `gameDate` is date-only; those rows are flagged `date_only` and are not treated as exact T-60 faceoff times. Odds `commence_time` at the snapshot is the start known to that quote.

## Data sources

See `docs/DATA_INVENTORY.md`. Defaults: NHL Web API (schedule/scores), MoneyPuck listed CSVs (non-commercial, credit required), local Odds-API extracts if present. No paid key is configured in this environment.

## Training / validation

Chronological walk-forward by season. Default historical split uses the last two seasons in the loaded table as validation and evaluation. Random splits are forbidden for primary evaluation.

## Computational budget

CPU-only. Core path: Elo + regularized logistic regression + optional Platt scaling on a validation window. Bootstrap 500 paired replicates. Power simulations 2000. Target runtime for the synthetic demo is well under 20 minutes.

## Primary metrics

1. Log loss on settled unique games  
2. Brier score  
3. Reliability bins  
4. Paired log-loss difference vs same-time de-vigged market, with block bootstrap  

Paper-trading unit returns are reported only when a timestamped quote exists; otherwise null.

## Success hierarchy

Engineering validity → data integrity → forecasting quality → uncertainty assessment → paper-trading evidence.

## Self-learning

Ingest → validate → score prior forecasts → fit challenger → compare to champion on predefined metrics → promote or retain → rollback if needed. Synthetic results cannot promote a real-data champion. Win streaks are not a promotion test.

## Out of scope

Real-money execution, sportsbook accounts, deposits, personalized wagering instructions, Kelly staking advice, and any bypass of gambling restrictions.

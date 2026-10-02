# NHL research platform

Point-in-time **forecasting and paper-trading research** for the 2026–27 NHL season. This repository tests whether reproducible predictive information exists. It does not place real bets and does not claim an edge over bookmakers.

The 2026–27 regular season began **2026-09-29** and is underway as of **2026-10-02**. Upcoming games must not be treated as completed.

## What is implemented

- Canonical games, odds snapshots, forecasts, and an append-only paper ledger
- Pregame Elo, rest, and rolling xG-share features with shift(1)
- Baselines: expanding home-win rate, Elo, same-book de-vigged moneyline
- Regularized logistic regression with fold-local scaling
- Block bootstrap, labeled intervals, and prospective power/MDE
- Champion / challenger promotion with rollback
- A local HTML dashboard and JSON exports
- Acceptance tests for leakage, settlement math, immutability, and synthetic labeling

This repository contains only the research platform. The previous betting-model scripts, pickled models, and vendored data lake are not part of this tree. Historical evaluation requires a fresh ingest; it is not reconstructed from the old project.

## Install

```bash
python3 -m pip install -e ".[dev]"
# or
python3 -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you later add an Odds API key. No key is required for the offline demo.

## Commands

```bash
python3 -m nhl_research.cli inventory
python3 -m nhl_research.cli demo          # labeled SYNTHETIC end-to-end run
python3 -m nhl_research.cli smoke         # chronological run if local historical files are present
python3 -m pytest tests/test_acceptance.py
```

After `demo` or `smoke`, open `reports/index.html`. Machine-readable forecasts: `reports/synthetic_forecasts.json` or `reports/real_smoke_forecasts.json`.

Unavailable quantities are JSON `null`. Research statuses are `RESEARCH_CANDIDATE`, `PASS` (criteria not met), and `BLOCKED`.

## Data terms

MoneyPuck listed files are used under the site’s **non-commercial** terms; credit [MoneyPuck.com](https://moneypuck.com/data.htm). NHL scores come from the public Web API. The Odds API historical endpoint requires a **paid** plan and was not called here.

## License

MIT for this software. Third-party data remain under their own terms.

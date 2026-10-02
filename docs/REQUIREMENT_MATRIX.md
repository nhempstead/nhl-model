# Requirement-to-implementation matrix

Statuses: IMPLEMENTED, TESTED, BLOCKED, NOT_YET_IMPLEMENTED.

| ID | Requirement | Status | Evidence |
|---|---|---|---|
| 4.1 | Project contract | IMPLEMENTED | docs/PROJECT_CONTRACT.md, configs/default.yaml |
| 4.2 | Point-in-time data foundation + manifests | IMPLEMENTED | nhl_research/data/*, warehouse manifests, docs/DATA_INVENTORY.md |
| 4.2b | Historical odds archive from Odds API live | BLOCKED | No ODDS_API_KEY; paid historical endpoint |
| 4.2c | NHL EDGE tracking | BLOCKED | No verified access |
| 4.3 | Feature registry + core features | IMPLEMENTED | nhl_research/features/registry.py, docs/FEATURE_REGISTRY.md |
| 4.3b | Pregame goalie scenarios | NOT_YET_IMPLEMENTED | Post-game starts excluded to avoid leakage |
| 4.4 | Baselines + logistic + market | IMPLEMENTED | nhl_research/models/* |
| 4.4b | Tree / hierarchical scores / totals joint | NOT_YET_IMPLEMENTED | Deferred until moneyline core is stable |
| 4.5 | Uncertainty, ESS, power | IMPLEMENTED / TESTED | nhl_research/uncertainty/*, tests |
| 4.6 | Odds math, de-vig, returns | IMPLEMENTED / TESTED | nhl_research/markets/*, tests |
| 4.7 | Chronological replay | IMPLEMENTED / TESTED | walk_forward; synthetic + optional real smoke |
| 4.7b | Closing-line value study | BLOCKED | No verified closing snapshots distinct from the T-60 quote |
| 4.8 | Promote / rollback | IMPLEMENTED / TESTED | learning/lifecycle.py |
| 4.9 | Dashboard + paper ledger | IMPLEMENTED | reports/index.html, ledger/paper.py; no wagering UI |
| 4.10 | Package, CLI, offline mode | IMPLEMENTED | nhl-research CLI, synthetic demo |
| 4.11 | Acceptance tests 1–12 | TESTED | tests/test_acceptance.py |
| 4.12 | Vertical slice | IMPLEMENTED | pipeline.run_synthetic_demo / run_real_smoke |

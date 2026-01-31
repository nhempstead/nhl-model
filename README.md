# NHL Betting Model

A quantitative NHL betting model built on real data.

## Status: Building

This model is under active development. Current phase: **Data Collection**

## Philosophy

- **CLV (Closing Line Value) is the only metric that matters**
- No fake edges, no made-up numbers
- Every prediction traces back to real data
- Walk-forward validation only (no leakage)
- Minimum 3% edge threshold to bet

## Data Requirements

| Dataset | Source | Rows | Status |
|---------|--------|------|--------|
| Game-level team stats | MoneyPuck | 226,851 | ✅ Complete |
| Shot-level xG data | MoneyPuck | 786,245 | ✅ Complete |
| Goalie season stats | MoneyPuck | 4,926 | ✅ Complete |
| Team season stats | MoneyPuck | 1,561 | ✅ Complete |
| Historical odds | TBD | ~10,000 | ⏳ Pending |

**Total: 1,019,583 rows of real data**

## Model Architecture

```
Features (pre-game only):
├── Team rolling stats (5/10/20 game windows)
│   ├── xGF, xGA, xG%
│   ├── Corsi%, Fenwick%
│   ├── Goals For/Against
│   └── Win %
├── Goalie stats (rolling)
│   ├── Save %
│   ├── GSAA
│   └── xGA vs actual
├── Schedule factors
│   ├── Rest days
│   ├── Travel distance
│   └── Back-to-back flag
└── Situational
    ├── Home/away splits
    └── Recent form (L5)

Target: Home win probability
Validation: Walk-forward (train→validate→test by season)
Calibration: Platt scaling on validation set
```

## Directories

```
data/
├── raw/          # Raw downloaded files
├── processed/    # Cleaned, feature-ready data
└── odds/         # Historical betting lines

models/
├── trained/      # Serialized models
└── evaluation/   # Backtest results

scripts/
├── collect/      # Data collection scripts
├── features/     # Feature engineering
├── train/        # Model training
└── predict/      # Live predictions
```

## Progress Log

- 2026-01-31 15:22: Data collection complete - 1M+ rows
  - Game-level team stats (2008-2025): 226,851 rows
  - Shot-level xG data (2018-2024): 786,245 rows
  - Goalie/team season summaries: 6,487 rows
- 2026-01-31 15:16: Repository created. Starting data collection.

## License

MIT

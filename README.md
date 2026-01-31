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

| Dataset | Source | Rows Needed | Status |
|---------|--------|-------------|--------|
| Historical games | NHL API | ~10,000 | 🔄 In Progress |
| Game-level xG | MoneyPuck | ~10,000 | ⏳ Pending |
| Historical odds | TBD | ~10,000 | ⏳ Pending |
| Goalie starts | NHL API | ~10,000 | ⏳ Pending |

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

- 2026-01-31: Repository created. Starting data collection.

## License

MIT

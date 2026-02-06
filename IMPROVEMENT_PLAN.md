# NHL Model V4 Improvement Plan
**Timeline: Feb 6-26, 2026 (Olympic Break)**
**Goal: Model probabilities on par with Pinnacle closing lines**

---

## Current State (V1 Model)

| Metric | Value | Target |
|--------|-------|--------|
| Test Brier | 0.2453 | <0.22 |
| Test Accuracy | 55.4% | >56% |
| Test LogLoss | 0.6836 | <0.65 |

Pinnacle's closing lines typically have Brier scores around 0.21-0.22 on NHL moneylines.

---

## Week 1 (Feb 6-12): Data & Features

### 1.1 Historical Closing Lines
- [ ] Acquire historical Pinnacle closing lines (2018-2025)
- [ ] Source options: oddsportal scrape, kaggle datasets, sbr historical
- [ ] Create benchmark: model vs Pinnacle closing line Brier score

### 1.2 Goalie Features
- [ ] Scrape historical starting goalie data (Daily Faceoff archives, NHL API)
- [ ] Create goalie metrics: GSAx, SV%, xGA, games played
- [ ] Feature: Starting goalie quality differential
- [ ] Feature: Backup vs starter indicator

### 1.3 Schedule Features  
- [ ] Calculate rest days for each team per game
- [ ] Back-to-back indicator
- [ ] Travel distance (arena coordinates)
- [ ] Time zone changes
- [ ] Home/road trip game number

### 1.4 Data Pipeline
- [ ] Build feature pipeline that can run daily
- [ ] Store pre-computed features for backtesting
- [ ] Validate no data leakage (all features pre-game only)

---

## Week 2 (Feb 13-19): Model Architecture

### 2.1 Baseline Calibration
- [ ] Calibrate V1 predictions against actual outcomes
- [ ] Measure calibration curve: predicted vs actual win rates by bucket
- [ ] Implement temperature scaling or Platt scaling properly

### 2.2 Ensemble Approach
- [ ] Train separate models:
  - XGBoost (team strength)
  - Logistic regression (interpretable)
  - Neural net (non-linear interactions)
- [ ] Stack or average predictions
- [ ] Cross-validate ensemble weights

### 2.3 Market Integration
- [ ] Experiment: use opening line as feature
- [ ] Experiment: predict closing line movement, not just outcome
- [ ] Key insight: we don't need to beat Pinnacle on all games, just find the 10% where they're off

### 2.4 Goalie-Specific Model
- [ ] Train sub-model for goalie confirmation scenarios
- [ ] Identify patterns: when does backup goalie create edge?
- [ ] Quantify typical line movement on goalie news

---

## Week 3 (Feb 20-26): Validation & Deployment

### 3.1 Walk-Forward Backtest
- [ ] Simulate betting 2024-25 season with V4 model
- [ ] Track hypothetical CLV against Pinnacle closing
- [ ] Require: positive CLV on >55% of bets
- [ ] Target: +2-3% average CLV

### 3.2 Edge Threshold Calibration
- [ ] Backtest different edge thresholds (2%, 3%, 5%, etc.)
- [ ] Find optimal threshold for ROI vs bet volume
- [ ] Implement Kelly sizing based on edge confidence

### 3.3 Production Pipeline
- [ ] Daily data refresh (AM)
- [ ] Prediction generation (before odds release)
- [ ] Edge calculation (after opening lines)
- [ ] Alert system for high-edge plays

### 3.4 Documentation
- [ ] Model card: features, training, validation
- [ ] Daily process checklist
- [ ] Failure modes and gotchas

---

## Success Criteria

1. **Brier Score <0.22** on held-out 2025 games
2. **Positive CLV** on simulated bets during backtest
3. **Calibration** - if model says 60%, win rate should be ~60%
4. **Edge identification** - find games where model disagrees with market by 3%+

---

## Data Sources Needed

| Source | Purpose | Status |
|--------|---------|--------|
| MoneyPuck | xG, Corsi, shot data | ✅ Have |
| NHL API | Schedules, rosters, results | ✅ Have |
| Daily Faceoff | Goalie confirmations | ✅ Have |
| The Odds API | Current odds | ✅ Have (limited) |
| Historical closing lines | Calibration target | ❌ Need |
| Historical goalie starts | Goalie features | ❌ Need |

---

## Key Principles

1. **CLV over win rate** - beating closing line is what matters
2. **Calibration over accuracy** - probabilities must mean what they say
3. **Pass most games** - only bet when edge is clear
4. **No shortcuts** - every number must trace to data

---

*Created: 2026-02-06*
*Review: Weekly progress check*

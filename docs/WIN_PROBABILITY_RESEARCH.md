# NHL Win Probability Models: Deep Research

## How Professional Models Calculate Win Percentages

### 1. MoneyPuck's Approach (Industry Standard)

MoneyPuck's model (rebuilt January 2025) breaks win probability into **three components**:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| Scoring Chances | 54% | xG, shot attempts, shots on goal, HDCF |
| Goaltending | 29% | GSAx, save %, recent performance |
| Ability to Win | 17% | Actual win % with recent weighting |

**Key Insights:**
- Less emphasis on recent games (prevents overreacting to slumps/streaks)
- Shooting talent adjustment (not all xG is equal)
- Home ice: ~54% historical win rate
- Back-to-back penalty: ~4% decrease in win probability

### 2. FiveThirtyEight Elo System

Uses classic Elo with NHL-specific adjustments:

**Base Formula:**
```
Pr(A wins) = 1 / (10^(-EloDiff/400) + 1)
```

**Adjustments:**
- **Home ice**: +50 Elo points (~57.1% win prob for equal teams)
- **Playoffs**: 1.25x multiplier on Elo difference
- **K-factor**: 6 (how much ratings change per game)
- **Margin of victory**: 0.6686 * ln(MOV) + 0.8048
- **Autocorrelation correction**: Prevents rating inflation for favorites

**Key Insight:** Elo is intentionally slow to react (K=6 is conservative). This prevents overweighting recent results but also means it can miss rapid team quality changes.

### 3. Evolving Hockey (Academic Approach)

Uses **Regularized Adjusted Plus-Minus (RAPM)** and **Goals Above Replacement (GAR)**:

- Player-level contributions aggregated to team strength
- Regression to mean for small samples
- Separate models for skaters vs goalies
- xG model trained on 50,000+ goals, 800,000+ shots

### 4. Sportsbook Line Setting (How Markets Work)

Professional books like Pinnacle:

1. **Opening line**: Set by proprietary model + historical data
2. **Market adjustment**: Lines move based on:
   - Sharp money (professional bettors)
   - Public money (recreational bettors)
   - Injury/lineup news
3. **Closing line**: Most accurate probability estimate (efficient market hypothesis)

**Key fact:** Closing lines at Pinnacle are ~98% efficient. Beating them consistently requires finding the 2-3% of mispriced games.

---

## What Our Model Was Doing Wrong

### Problem 1: Over-Compressed Predictions
- Model predicted 45-55% for almost all games
- Real win probabilities range from 30-70%
- Caused by: High regularization (reg_alpha=1.0, reg_lambda=1.0)

### Problem 2: No Home Ice Adjustment
- Model predicted 49.5% home win rate
- Actual NHL home win rate: 51-54%
- Created systematic bias toward road underdogs

### Problem 3: Too Much Recency Weighting
- Original windows: L5, L10, L20 games
- Slumps/streaks dominated predictions
- MoneyPuck explicitly reduced recency influence in 2025 rebuild

---

## Recommended Model Architecture

Based on research, a proper NHL win probability model should:

### Features (in order of importance):
1. **Season-long xG differential** (40+ games)
2. **Goaltender quality** (GSAx over 2+ seasons, regressed)
3. **Home/away split** (+3-4% for home team)
4. **Rest advantage** (back-to-back = -4%)
5. **Recent form** (L10 games, weighted at ~20%)
6. **Special teams** (PP%, PK%)
7. **Score effects** (leading team plays defensively)

### Calibration:
- Predictions should match actual outcomes across all buckets
- 60% predictions should win 60% of the time
- Use isotonic regression or Platt scaling

### What NOT to Do:
- Don't overweight last 5-10 games
- Don't ignore goaltender (29% of outcome per MoneyPuck)
- Don't compress all predictions to 45-55%
- Don't ignore home ice advantage

---

## Action Items for V2 Model

1. **Add explicit goaltender features**
   - Starting goalie GSAx
   - Recent save %
   - Career playoff performance

2. **Reduce recency bias further**
   - Weight: 20% last 10 games, 80% season-long
   - Or use Elo-style decay (K=6)

3. **Verify calibration**
   - Plot reliability diagram
   - Ensure 40% predictions win 40%, etc.

4. **Add special teams**
   - PP% differential
   - PK% differential

5. **Consider Elo hybrid**
   - Elo for team strength baseline
   - xG features for game-specific adjustment

---

## Sources

- MoneyPuck.com - About page (January 2025 model rebuild)
- FiveThirtyEight - "How Our NHL Predictions Work"
- Evolving Hockey - Model methodology
- College Football Data blog - Elo implementation details

*Research compiled: 2026-02-04*

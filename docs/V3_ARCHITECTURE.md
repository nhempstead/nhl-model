# NHL Model V3: Beat The Books

## The Goal
Build a model that generates consistent CLV against sharp books (Pinnacle).

## Where Books Are Beatable

### 1. Information Speed
- Goalie announcements: 10-60 min window before lines adjust
- Injury news: Beat reporters tweet before official
- Lineup changes: Morning skate information

### 2. Goaltender Modeling
- Books use basic starter/backup splits
- We can model: GSAx trends, fatigue, matchup history, recent workload

### 3. Schedule/Fatigue
- Travel distance + time zones
- Back-to-back performance decay
- 3-in-4, 4-in-6 situations
- Altitude effects (Denver, etc.)

### 4. Market Overreactions
- Public overweights recent blowouts
- Books shade against public favorites
- Regression candidates (lucky/unlucky teams)

---

## V3 Model Architecture

### Layer 1: Team Strength (Elo Baseline)
```
Base Elo Rating per team (carries season-to-season)
- K-factor: 6 (slow, stable)
- Home ice: +50 Elo
- Margin of victory adjustment
- Playoff multiplier
```

### Layer 2: Expected Goals Model
```
Season xG differential (primary signal)
- 5v5 xGF/60 and xGA/60
- Score-adjusted (SAXG)
- Quality of competition adjustment

Recent form adjustment (20% weight max)
- Last 10 games xG differential
- Trend detection (improving/declining)
```

### Layer 3: Goaltender Model (29% of outcome per MoneyPuck)
```
For each starting goalie:
- Career GSAx (regressed to mean)
- Last 30 games GSAx
- Workload factor (games in last 7/14 days)
- Home/away splits
- Back-to-back penalty
- Matchup history vs opponent

Uncertainty adjustment:
- Backup goalies: wider confidence interval
- Unconfirmed starter: blend probabilities
```

### Layer 4: Special Teams
```
Power play efficiency differential
- PP% vs opponent PK%
- PP opportunities per game

Penalty kill differential  
- PK% vs opponent PP%
- Penalty differential (discipline)
```

### Layer 5: Situational Adjustments
```
Schedule spots:
- Back-to-back: -4% win prob
- Travel (cross-country): -1%
- Altitude (at Colorado): -1%
- 3-in-4 nights: -2%
- Rest advantage (2+ days vs 1): +2%

Motivation factors:
- Playoff race position
- Revenge game (lost badly recently)
- Division rival
```

### Layer 6: Calibration
```
Isotonic regression on validation set
Ensure:
- 40% predictions win 40%
- 60% predictions win 60%
- Full probability range (0.30 - 0.70)
```

---

## Data Requirements

### Must Have
- [ ] Real-time starting goalie confirmations (Daily Faceoff API or scraper)
- [ ] Goalie GSAx historical (Evolving Hockey or MoneyPuck)
- [ ] Team xG data (MoneyPuck - need to fix 403)
- [ ] Live odds feed (The Odds API - have this)
- [ ] Schedule with times (NHL API - have this)

### Nice to Have
- [ ] Player-level xG contributions
- [ ] Line combinations
- [ ] Injury reports with severity
- [ ] Travel distance calculator
- [ ] Historical betting line data (for CLV)

---

## Execution Strategy

### Pre-Game (Morning)
1. Scrape projected starters
2. Run model with projected lineups
3. Identify potential edges
4. Set alerts for goalie confirmations

### Goalie Confirmation Window (10 AM - 5 PM)
1. Monitor for official announcements
2. Re-run model immediately on confirmation
3. Compare to current odds
4. Execute bets within 15 min if edge > 3%

### Line Shopping
1. Check odds at 5+ books
2. Calculate no-vig probability from Pinnacle
3. Find best price for our side
4. Track all bets with opening/closing line

---

## Success Metrics

### Primary: CLV (Closing Line Value)
```
CLV = (Our_Odds / Closing_Odds) - 1
Target: +2% average CLV
```

### Secondary: ROI
```
Target: 3-5% ROI over 500+ bets
Note: ROI is noisy short-term, CLV is better signal
```

### Calibration
```
Brier score < 0.24
Log loss < 0.68
Reliability diagram should be diagonal
```

---

## Implementation Plan

### Phase 1: Fix Data Pipeline (This Week)
- [ ] Fix MoneyPuck scraper (bypass 403)
- [ ] Add goalie GSAx data source
- [ ] Add schedule fatigue features

### Phase 2: Rebuild Model (Next Week)
- [ ] Implement Elo layer
- [ ] Add goaltender model
- [ ] Add special teams
- [ ] Add situational adjustments
- [ ] Proper calibration

### Phase 3: Execution System (Week 3)
- [ ] Goalie confirmation alerts
- [ ] Auto line comparison
- [ ] Bet logging with timestamps
- [ ] CLV tracking

### Phase 4: Iteration (Ongoing)
- [ ] Weekly model review
- [ ] Feature importance analysis
- [ ] A/B test new features
- [ ] Track by bet type (favorite/dog, home/away)

---

## The Edge

Books are good at:
- Processing public information efficiently
- Balancing action
- Setting accurate closing lines

Books are NOT as good at:
- Goaltender modeling at granular level
- Speed on breaking news
- Niche schedule spots
- Small market games early in day

We win by:
1. Better goalie model
2. Faster execution on news
3. Discipline (only bet real edges)
4. Track everything, learn constantly

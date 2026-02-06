# 3-Week Plan: Beat Pinnacle

**Timeline:** Feb 6-26, 2026 (Olympic Break)
**Goal:** Model Brier < 0.2248 (beat market)
**Current:** Model Brier 0.2414 (7.4% worse)

---

## Week 1: Data & Feature Engineering (Feb 6-12)

### Day 1-2: Historical Goalie Starts
**Problem:** Using season-level goalie ratings, market uses game-day confirmation
**Solution:** Scrape historical starting goalie data

- [ ] Scrape Daily Faceoff archives for 2020-2025 starting goalies
- [ ] Build game-level goalie features (who actually started, not team average)
- [ ] Calculate "backup start" indicator (huge edge when market doesn't adjust)
- [ ] Add goalie rest days (G on B2B is different than skater B2B)

**Expected Impact:** +1-2% Brier improvement

### Day 3-4: Line Movement Features
**Problem:** We use static closing odds, market moves on information
**Solution:** Capture line movement as a feature

- [ ] Fetch opening lines from The Odds API (we have budget)
- [ ] Calculate: `line_move = closing_prob - opening_prob`
- [ ] Reverse line movement indicator (public vs sharp)
- [ ] Steam move detector (rapid movement = sharp action)

**Expected Impact:** +0.5-1% Brier improvement

### Day 5-6: Situational Features
**Problem:** Missing game context that sharps use
**Solution:** Add situational edges

- [ ] Playoff implications (clinched, eliminated, fighting for spot)
- [ ] Revenge games (recent head-to-head blowout losses)
- [ ] Division games (more competitive historically)
- [ ] Home stand / road trip game number
- [ ] Coming off long break (All-Star, bye week)
- [ ] Altitude adjustment (Denver games)

**Expected Impact:** +0.5% Brier improvement

### Day 7: Integration & Retrain
- [ ] Merge all new features into training data
- [ ] Retrain V3 model with expanded feature set
- [ ] Benchmark against market
- [ ] Document feature importance changes

**Week 1 Target:** Brier < 0.235

---

## Week 2: Model Architecture (Feb 13-19)

### Day 8-9: Ensemble Methods
**Problem:** Single XGBoost may miss patterns other models catch
**Solution:** Build ensemble of diverse models

- [ ] Train 5 base models:
  1. XGBoost (current)
  2. LightGBM (faster, different splits)
  3. CatBoost (better with categoricals)
  4. Logistic Regression (linear baseline)
  5. Neural Network (non-linear interactions)
- [ ] Stack with meta-learner
- [ ] Cross-validate ensemble weights

**Expected Impact:** +1-2% Brier improvement

### Day 10-11: Calibration Deep Dive
**Problem:** Model probabilities don't mean what they say
**Solution:** Advanced calibration techniques

- [ ] Platt scaling vs Isotonic vs Beta calibration
- [ ] Calibration by probability bucket analysis
- [ ] Temperature scaling for confidence adjustment
- [ ] Build calibration curves, target flat residuals

**Expected Impact:** +0.5-1% Brier improvement

### Day 12-13: Market-Aware Training
**Problem:** Training to predict outcomes, not beat market
**Solution:** Train against market-relative targets

- [ ] Target = `actual_outcome - market_prob` (beat market, not predict game)
- [ ] Weight recent games higher (market evolves)
- [ ] Feature: `market_home_prob` as input (learn when market is wrong)
- [ ] Identify market blind spots (B2B not fully priced, backup goalies)

**Expected Impact:** +1% Brier improvement on market-overlap games

### Day 14: Full Retrain & Benchmark
- [ ] Combine ensemble + calibration
- [ ] Full backtest against 2020-2025 market odds
- [ ] Calculate CLV on hypothetical bets
- [ ] Document what's working vs not

**Week 2 Target:** Brier < 0.225 (match market)

---

## Week 3: Edge Detection & Validation (Feb 20-26)

### Day 15-16: Where Does Market Get It Wrong?
**Problem:** Need to find systematic market inefficiencies
**Solution:** Deep analysis of model vs market disagreements

- [ ] Identify games where model >> market (high-edge candidates)
- [ ] Cluster these games: what patterns emerge?
- [ ] Backtest: do high-edge predictions actually win more?
- [ ] Build edge confidence score

Key hypotheses to test:
1. Backup goalie not fully priced (15-30 min window)
2. B2B home team undervalued by market
3. Travel fatigue (3+ timezone changes)
4. Early season (market slow to update priors)

**Expected Impact:** Edge identification, not Brier improvement

### Day 17-18: Walk-Forward Validation
**Problem:** Need to prove model works in real-time, not just backtest
**Solution:** Strict walk-forward testing

- [ ] Train on 2020-2023, test on 2024 month-by-month
- [ ] No lookahead: only use data available at prediction time
- [ ] Track simulated CLV for each month
- [ ] Identify any seasonal degradation

**Requirement:** Positive CLV in 8+ of 12 months

### Day 19-20: Production Pipeline
**Problem:** Model needs to run daily with fresh data
**Solution:** Build automated prediction pipeline

- [ ] Morning: Fetch overnight data (injuries, transactions)
- [ ] Pre-game: Get confirmed goalies, update features
- [ ] Generate predictions with confidence intervals
- [ ] Compare to market, flag high-edge opportunities
- [ ] Evening: Record outcomes, track CLV

### Day 21: Final Model & Documentation
- [ ] Lock final model weights
- [ ] Document all features, their importance, rationale
- [ ] Create model card (training data, validation results, limitations)
- [ ] Set betting thresholds (edge % required to bet)
- [ ] Prepare for Feb 26 NHL return

**Week 3 Target:** 
- Brier < 0.224 (beat market)
- Positive backtested CLV
- Production-ready pipeline

---

## Success Metrics

| Metric | Current | Week 1 | Week 2 | Week 3 |
|--------|---------|--------|--------|--------|
| Test Brier | 0.2414 | <0.235 | <0.225 | <0.224 |
| Gap to Market | +7.4% | +4% | ±0% | -1% |
| CLV (backtest) | N/A | N/A | Track | >+2% |
| Edge Detection | None | Basic | Refined | Production |

## Resources Needed

- **The Odds API:** ~10K more credits for opening lines (have 4.5K remaining, may need upgrade)
- **Daily Faceoff:** Scrape historical goalie starts (free)
- **Compute:** Local training is fine, ~30 min per full retrain

## Risk Factors

1. **Overfitting:** More features = more risk. Cross-validate everything.
2. **Data leakage:** Ensure no future information in features.
3. **Market evolution:** 2020 patterns may not hold in 2026.
4. **Sample size:** Some edges may be noise with 1,400 games/year.

## Daily Check-In

Each day:
1. What was built/tested?
2. Did it improve the benchmark?
3. What's blocking progress?
4. Next day's priority

---

**Bottom Line:** Beat Pinnacle by finding what they miss:
- Real-time goalie confirmation
- Travel/fatigue not fully priced
- Backup starts = market adjustment lag
- Compound multiple small edges into consistent CLV

*Created: 2026-02-06*

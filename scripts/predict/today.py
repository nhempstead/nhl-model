#!/usr/bin/env python3
"""
Generate predictions for today's games using the trained model.
"""

import pandas as pd
import numpy as np
import pickle
import requests
import os
from datetime import datetime, date

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models/trained')

def load_model():
    """Load trained model"""
    path = os.path.join(MODEL_DIR, 'model_v1.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_todays_games():
    """Fetch today's NHL games"""
    today = date.today().strftime('%Y-%m-%d')
    url = f"https://api-web.nhle.com/v1/score/{today}"
    
    resp = requests.get(url, timeout=10)
    data = resp.json()
    
    games = []
    for g in data.get('games', []):
        if g.get('gameType') == 2:  # Regular season
            games.append({
                'game_id': g['id'],
                'home_team': g['homeTeam']['abbrev'],
                'away_team': g['awayTeam']['abbrev'],
                'start_time': g.get('startTimeUTC'),
                'game_state': g.get('gameState'),
            })
    
    return games

def get_team_features(team, game_data):
    """Get rolling features for a team from historical data"""
    
    # Load the processed matchups to get recent team stats
    path = os.path.join(DATA_DIR, 'raw/moneypuck_all_games.csv')
    df = pd.read_csv(path)
    
    # Filter to 5on5 and this team
    df = df[(df['situation'] == '5on5') & (df['team'] == team)].copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'].astype(str), format='%Y%m%d')
    df = df.sort_values('gameDate')
    
    # Get last 20 games
    recent = df.tail(20)
    
    if len(recent) < 5:
        return None
    
    # Compute rolling averages
    features = {}
    
    # 5-game rolling
    last5 = recent.tail(5)
    features['xGoalsFor_L5'] = last5['xGoalsFor'].mean()
    features['xGoalsAgainst_L5'] = last5['xGoalsAgainst'].mean()
    features['corsiPercentage_L5'] = last5['corsiPercentage'].mean()
    features['fenwickPercentage_L5'] = last5['fenwickPercentage'].mean()
    features['xGoalsPercentage_L5'] = last5['xGoalsPercentage'].mean()
    
    # 10-game rolling
    last10 = recent.tail(10)
    features['xGoalsFor_L10'] = last10['xGoalsFor'].mean()
    features['xGoalsAgainst_L10'] = last10['xGoalsAgainst'].mean()
    features['corsiPercentage_L10'] = last10['corsiPercentage'].mean()
    features['fenwickPercentage_L10'] = last10['fenwickPercentage'].mean()
    features['xGoalsPercentage_L10'] = last10['xGoalsPercentage'].mean()
    features['highDangerShotsFor_L10'] = last10['highDangerShotsFor'].mean() if 'highDangerShotsFor' in last10.columns else 0
    features['highDangerShotsAgainst_L10'] = last10['highDangerShotsAgainst'].mean() if 'highDangerShotsAgainst' in last10.columns else 0
    
    # Win rate
    if 'goalsFor' in df.columns and 'goalsAgainst' in df.columns:
        last10['win'] = (last10['goalsFor'] > last10['goalsAgainst']).astype(int)
        features['winPct_L10'] = last10['win'].mean()
        last20 = recent.tail(20)
        last20['win'] = (last20['goalsFor'] > last20['goalsAgainst']).astype(int)
        features['winPct_L20'] = last20['win'].mean()
    
    # 20-game rolling
    features['xGoalsFor_L20'] = recent['xGoalsFor'].mean()
    features['xGoalsAgainst_L20'] = recent['xGoalsAgainst'].mean()
    features['corsiPercentage_L20'] = recent['corsiPercentage'].mean()
    features['fenwickPercentage_L20'] = recent['fenwickPercentage'].mean()
    features['xGoalsPercentage_L20'] = recent['xGoalsPercentage'].mean()
    
    # Rest days
    if len(df) >= 2:
        last_game = df['gameDate'].iloc[-1]
        today = pd.Timestamp.now()
        features['rest_days'] = min(7, max(1, (today - last_game).days))
    else:
        features['rest_days'] = 3
    
    return features

def build_matchup_features(home_team, away_team, model_package):
    """Build feature vector for a matchup"""
    
    home_f = get_team_features(home_team, None)
    away_f = get_team_features(away_team, None)
    
    if home_f is None or away_f is None:
        return None
    
    # Build feature dict
    features = {}
    
    # Home features
    for k, v in home_f.items():
        features[f'h_{k}'] = v
    
    # Away features
    for k, v in away_f.items():
        features[f'a_{k}'] = v
    
    # Differentials
    features['xGoalsPercentage_L10_diff'] = home_f.get('xGoalsPercentage_L10', 50) - away_f.get('xGoalsPercentage_L10', 50)
    features['corsiPercentage_L10_diff'] = home_f.get('corsiPercentage_L10', 50) - away_f.get('corsiPercentage_L10', 50)
    features['xGoalsFor_L10_diff'] = home_f.get('xGoalsFor_L10', 0) - away_f.get('xGoalsFor_L10', 0)
    features['xGoalsAgainst_L10_diff'] = home_f.get('xGoalsAgainst_L10', 0) - away_f.get('xGoalsAgainst_L10', 0)
    features['winPct_L10_diff'] = home_f.get('winPct_L10', 0.5) - away_f.get('winPct_L10', 0.5)
    
    features['h_rest_days'] = home_f.get('rest_days', 3)
    features['a_rest_days'] = away_f.get('rest_days', 3)
    features['rest_diff'] = features['h_rest_days'] - features['a_rest_days']
    
    return features

def predict_game(home_team, away_team, model_package):
    """Generate prediction for a single game"""
    
    features = build_matchup_features(home_team, away_team, model_package)
    if features is None:
        return None
    
    # Build feature vector in correct order
    feature_cols = model_package['feature_cols']
    X = pd.DataFrame([features])
    
    # Fill missing columns with 0
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    
    X = X[feature_cols].fillna(0)
    
    # Get raw prediction
    model = model_package['model']
    calibrator = model_package['calibrator']
    
    raw_prob = model.predict_proba(X)[0, 1]
    cal_prob = calibrator.predict_proba([[raw_prob]])[0, 1]
    
    return {
        'home_prob': cal_prob,
        'away_prob': 1 - cal_prob,
        'raw_prob': raw_prob,
    }

def get_current_odds():
    """Fetch current odds from The Odds API"""
    api_key = os.environ.get('ODDS_API_KEY', 'b181ec65384e338495a1b5b5fe1ee30b')
    url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
    
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
    }
    
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        print(f"Odds API error: {resp.status_code}")
        return {}
    
    data = resp.json()
    odds_map = {}
    
    # Team name mapping (API names to abbreviations)
    name_to_abbrev = {
        'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
        'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
        'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
        'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
        'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
        'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD',
        'New York Islanders': 'NYI', 'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT',
        'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS',
        'Seattle Kraken': 'SEA', 'St Louis Blues': 'STL', 'St. Louis Blues': 'STL',
        'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR', 'Utah Hockey Club': 'UTA',
        'Vancouver Canucks': 'VAN', 'Vegas Golden Knights': 'VGK', 'Washington Capitals': 'WSH',
        'Winnipeg Jets': 'WPG',
    }
    
    for game in data:
        home_full = game.get('home_team', '')
        away_full = game.get('away_team', '')
        
        home = name_to_abbrev.get(home_full, home_full)
        away = name_to_abbrev.get(away_full, away_full)
        
        key = f"{away}@{home}"
        odds_map[key] = {'home_team': home, 'away_team': away}
        
        # Get consensus odds (first bookmaker)
        for bm in game.get('bookmakers', []):
            for market in bm.get('markets', []):
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        team_full = outcome['name']
                        team = name_to_abbrev.get(team_full, team_full)
                        price = outcome['price']
                        
                        if team == home:
                            odds_map[key]['home_odds'] = price
                        else:
                            odds_map[key]['away_odds'] = price
            break  # Use first bookmaker only
    
    return odds_map

def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def main():
    print("="*70)
    print(f"NHL MODEL V2 PREDICTIONS - {date.today()}")
    print("="*70)
    
    # Load model
    model = load_model()
    print(f"Model loaded: {len(model['feature_cols'])} features")
    print(f"Test accuracy: {model['metrics']['test']['test_accuracy']:.1%}")
    
    # Get today's games
    games = get_todays_games()
    print(f"\nFound {len(games)} games today")
    
    # Get current odds
    odds_map = get_current_odds()
    
    # Generate predictions
    print("\n" + "-"*70)
    
    results = []
    for game in games:
        home = game['home_team']
        away = game['away_team']
        
        pred = predict_game(home, away, model)
        
        if pred is None:
            print(f"{away} @ {home}: Insufficient data")
            continue
        
        # Find odds - use exact key match
        game_key = f"{away}@{home}"
        odds_data = odds_map.get(game_key, {})
        
        home_odds = odds_data.get('home_odds')
        away_odds = odds_data.get('away_odds')
        
        # Calculate edge
        result = {
            'game': f"{away} @ {home}",
            'home_team': home,
            'away_team': away,
            'model_home': pred['home_prob'],
            'model_away': pred['away_prob'],
        }
        
        if home_odds:
            market_home = american_to_prob(home_odds)
            result['home_odds'] = home_odds
            result['market_home'] = market_home
            result['home_edge'] = (pred['home_prob'] - market_home) * 100
        
        if away_odds:
            market_away = american_to_prob(away_odds)
            result['away_odds'] = away_odds
            result['market_away'] = market_away
            result['away_edge'] = (pred['away_prob'] - market_away) * 100
        
        results.append(result)
        
        # Print
        print(f"\n{away} @ {home}")
        print(f"  Model: {home} {pred['home_prob']:.1%} | {away} {pred['away_prob']:.1%}")
        
        if home_odds and away_odds:
            print(f"  Odds:  {home} ({home_odds:+d}) | {away} ({away_odds:+d})")
            print(f"  Edge:  {home} {result.get('home_edge', 0):+.1f}% | {away} {result.get('away_edge', 0):+.1f}%")
            
            # Flag value plays
            if result.get('home_edge', 0) > 3:
                print(f"  >>> VALUE: {home} {home_odds:+d} ({result['home_edge']:+.1f}% edge)")
            elif result.get('away_edge', 0) > 3:
                print(f"  >>> VALUE: {away} {away_odds:+d} ({result['away_edge']:+.1f}% edge)")
    
    # Summary
    print("\n" + "="*70)
    print("VALUE PLAYS (3%+ edge)")
    print("="*70)
    
    value_plays = []
    for r in results:
        if r.get('home_edge', 0) > 3:
            value_plays.append({
                'game': r['game'],
                'pick': r['home_team'],
                'odds': r.get('home_odds'),
                'model': r['model_home'],
                'market': r.get('market_home'),
                'edge': r['home_edge'],
            })
        elif r.get('away_edge', 0) > 3:
            value_plays.append({
                'game': r['game'],
                'pick': r['away_team'],
                'odds': r.get('away_odds'),
                'model': r['model_away'],
                'market': r.get('market_away'),
                'edge': r['away_edge'],
            })
    
    if value_plays:
        for vp in sorted(value_plays, key=lambda x: -x['edge']):
            print(f"{vp['game']}: {vp['pick']} ({vp['odds']:+d})")
            print(f"  Model: {vp['model']:.1%} | Market: {vp['market']:.1%} | Edge: {vp['edge']:+.1f}%")
    else:
        print("No value plays found with 3%+ edge")
    
    return results


if __name__ == '__main__':
    main()

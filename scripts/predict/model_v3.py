#!/usr/bin/env python3
"""
NHL Model V3 - Current-State Prediction Engine

Predicts win probabilities based on CURRENT data:
1. Today's power ratings (from MoneyPuck xG)
2. Confirmed starting goalie
3. Key player availability
4. Home ice advantage
5. Rest/schedule factors

This replaces the stale rolling-average approach with live data.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import requests

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / 'data'


def load_power_ratings() -> dict:
    """Load current power ratings from daily refresh."""
    ratings_file = DATA_DIR / 'ratings' / 'power_ratings.json'
    
    if not ratings_file.exists():
        print("⚠️  No power ratings found. Run daily_refresh.py first.")
        return {}
    
    with open(ratings_file) as f:
        data = json.load(f)
    
    return data.get('ratings', {})


def load_injuries() -> dict:
    """Load current injuries from Daily Faceoff data."""
    injury_file = DATA_DIR / 'rosters' / 'daily_faceoff_rosters.json'
    
    if not injury_file.exists():
        return {}
    
    with open(injury_file) as f:
        data = json.load(f)
    
    injuries = {}
    for team, roster in data.get('teams', {}).items():
        if 'injuries' in roster:
            injuries[team] = roster['injuries']
    
    return injuries


def load_rosters() -> dict:
    """Load current NHL rosters."""
    roster_file = DATA_DIR / 'rosters' / 'nhl_rosters.json'
    
    if not roster_file.exists():
        return {}
    
    with open(roster_file) as f:
        data = json.load(f)
    
    return data.get('teams', {})


def load_goalie_stats() -> pd.DataFrame:
    """Load current season goalie stats."""
    goalie_file = DATA_DIR / 'raw' / 'moneypuck_goalies_current.csv'
    
    if not goalie_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(goalie_file)


def get_goalie_rating(team: str, goalie_stats: pd.DataFrame, injuries: dict) -> tuple:
    """
    Get goalie rating for a team, accounting for injuries.
    Returns (goalie_name, rating, is_backup)
    
    Rating scale: 0 = replacement level, 100 = elite
    Based on: save percentage and goals saved above expected (GSAx)
    """
    team_injuries = injuries.get(team, [])
    injured_names = {inj['name'].lower() for inj in team_injuries}
    
    # Get team's goalies from stats (filter to 'all' situation)
    if goalie_stats.empty or 'team' not in goalie_stats.columns:
        return ("Unknown", 50, False)
    
    team_goalies = goalie_stats[
        (goalie_stats['team'] == team) & 
        (goalie_stats['situation'] == 'all')
    ].copy()
    
    if team_goalies.empty:
        # Try without situation filter
        team_goalies = goalie_stats[goalie_stats['team'] == team].copy()
    
    if team_goalies.empty:
        return ("Unknown", 50, False)
    
    # Sort by games played to identify starter
    if 'games_played' in team_goalies.columns:
        team_goalies = team_goalies.sort_values('games_played', ascending=False)
    
    def calculate_goalie_rating(row):
        """Calculate rating from MoneyPuck goalie data.
        
        Rating scale (0-100):
        - 0-20: Poor (backup/AHL level)
        - 20-40: Below average
        - 40-60: Average starter
        - 60-80: Good starter
        - 80-100: Elite
        
        Based on save percentage and GSAx (goals saved above expected).
        """
        goals = row.get('goals', 0) or 0
        shots = row.get('ongoal', 0) or 0
        xgoals = row.get('xGoals', 0) or 0
        games = row.get('games_played', 1) or 1
        
        if shots > 0:
            sv_pct = 1 - (goals / shots)
        else:
            sv_pct = 0.900
        
        # GSAx = expected goals - actual goals (positive = good)
        gsax = xgoals - goals
        gsax_per_game = gsax / games
        
        # Rating formula - recalibrated:
        # Base 50 at .900 SV%, +20 per .010 above, -15 per .010 below
        if sv_pct >= 0.900:
            sv_rating = 50 + (sv_pct - 0.900) * 2000  # Elite: .920 = 90
        else:
            sv_rating = 50 + (sv_pct - 0.900) * 1500  # Bad: .880 = 20
        
        # GSAx bonus: +2 per goal saved above expected
        gsax_bonus = gsax * 2
        
        # Combine (weight SV% more heavily for small sample sizes)
        if games < 10:
            rating = sv_rating * 0.8 + gsax_bonus * 0.2
        else:
            rating = sv_rating * 0.6 + gsax_bonus * 0.4 + 20  # Add baseline
        
        return max(10, min(95, rating)), sv_pct, gsax
    
    # Get primary goalie
    primary = team_goalies.iloc[0]
    primary_name = primary.get('name', 'Unknown')
    primary_rating, primary_sv, primary_gsax = calculate_goalie_rating(primary)
    
    # Check if primary goalie is injured
    if primary_name.lower() in injured_names:
        # Use backup
        if len(team_goalies) > 1:
            backup = team_goalies.iloc[1]
            backup_name = backup.get('name', 'Unknown')
            backup_rating, _, _ = calculate_goalie_rating(backup)
            return (backup_name, backup_rating, True)
        else:
            # Unknown backup - assume replacement level
            return ("Backup", 25, True)
    
    return (primary_name, primary_rating, False)


def calculate_roster_adjustment(team: str, injuries: dict) -> float:
    """
    Calculate win probability adjustment for missing key players.
    Returns adjustment in percentage points.
    
    Key impact players:
    - Elite goalie (Shesterkin, Vasilevskiy, etc.): -10 to -15%
    - Top defenseman (Fox, Makar, etc.): -3 to -5%
    - Elite forward (McDavid, Matthews, etc.): -2 to -4%
    """
    adjustment = 0.0
    team_injuries = injuries.get(team, [])
    
    # Known elite players and their impact
    ELITE_GOALIES = {
        'igor shesterkin': -12,
        'andrei vasilevskiy': -10,
        'connor hellebuyck': -10,
        'ilya sorokin': -8,
        'juuse saros': -8,
        'thatcher demko': -7,
    }
    
    ELITE_DEFENSEMEN = {
        'adam fox': -5,
        'cale makar': -5,
        'quinn hughes': -4,
        'victor hedman': -4,
        'charlie mcavoy': -4,
        'miro heiskanen': -4,
    }
    
    ELITE_FORWARDS = {
        'connor mcdavid': -5,
        'auston matthews': -4,
        'leon draisaitl': -4,
        'nathan mackinnon': -4,
        'nikita kucherov': -4,
        'artemi panarin': -3,
        'david pastrnak': -3,
        'mikko rantanen': -3,
    }
    
    for inj in team_injuries:
        name = inj.get('name', '').lower()
        
        if name in ELITE_GOALIES:
            adjustment += ELITE_GOALIES[name]
        elif name in ELITE_DEFENSEMEN:
            adjustment += ELITE_DEFENSEMEN[name]
        elif name in ELITE_FORWARDS:
            adjustment += ELITE_FORWARDS[name]
    
    return adjustment


def predict_game(home_team: str, away_team: str,
                 power_ratings: dict, 
                 goalie_stats: pd.DataFrame,
                 injuries: dict) -> dict:
    """
    Predict win probability for a single game.
    
    Method:
    1. Start with power rating differential
    2. Add home ice advantage (~3-4%)
    3. Adjust for goalie matchup
    4. Adjust for key missing players
    """
    
    # Get power ratings (default to 100 if not found)
    home_power = power_ratings.get(home_team, {}).get('power_rating', 100)
    away_power = power_ratings.get(away_team, {}).get('power_rating', 100)
    
    # Power differential -> base win probability
    # Each point of power rating ~ 1% win probability shift
    power_diff = home_power - away_power
    
    # Start at 50%, adjust by power differential
    # Cap the adjustment so extreme ratings don't produce unrealistic probs
    base_prob = 0.50 + (power_diff / 100) * 0.10  # 10 point diff = 10% swing
    
    # Home ice advantage: +3.5%
    home_ice = 0.035
    
    # Goalie adjustment
    home_goalie, home_goalie_rating, home_backup = get_goalie_rating(home_team, goalie_stats, injuries)
    away_goalie, away_goalie_rating, away_backup = get_goalie_rating(away_team, goalie_stats, injuries)
    
    # Goalie rating differential -> adjustment
    # Each 10 points of goalie rating ~ 1% win probability
    goalie_diff = home_goalie_rating - away_goalie_rating
    goalie_adj = goalie_diff / 1000  # Max ~5% swing for 50-point diff
    
    # Roster adjustments for missing key players
    home_roster_adj = calculate_roster_adjustment(home_team, injuries) / 100
    away_roster_adj = calculate_roster_adjustment(away_team, injuries) / 100
    
    # Net roster impact (if home team missing players, it hurts them)
    roster_adj = home_roster_adj - away_roster_adj
    
    # Final probability
    home_prob = base_prob + home_ice + goalie_adj + roster_adj
    
    # Ensure probability is in valid range
    home_prob = max(0.15, min(0.85, home_prob))
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_prob': round(home_prob, 3),
        'away_prob': round(1 - home_prob, 3),
        'components': {
            'base_prob': round(base_prob, 3),
            'home_ice': round(home_ice, 3),
            'goalie_adj': round(goalie_adj, 3),
            'roster_adj': round(roster_adj, 3),
        },
        'home_power': home_power,
        'away_power': away_power,
        'home_goalie': home_goalie,
        'away_goalie': away_goalie,
        'home_goalie_rating': round(home_goalie_rating, 1),
        'away_goalie_rating': round(away_goalie_rating, 1),
        'home_backup': home_backup,
        'away_backup': away_backup,
    }


def get_todays_games():
    """Fetch today's NHL games."""
    today = date.today().strftime('%Y-%m-%d')
    url = f"https://api-web.nhle.com/v1/score/{today}"
    
    try:
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
    except Exception as e:
        print(f"Error fetching games: {e}")
        return []


def get_current_odds():
    """Fetch current odds from The Odds API."""
    api_key = os.environ.get('ODDS_API_KEY', 'b181ec65384e338495a1b5b5fe1ee30b')
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds/"
    
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return {}
        
        data = resp.json()
        odds_map = {}
        
        name_to_abbrev = {
            'Anaheim Ducks': 'ANA', 'Boston Bruins': 'BOS', 'Buffalo Sabres': 'BUF',
            'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR', 'Chicago Blackhawks': 'CHI',
            'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ', 'Dallas Stars': 'DAL',
            'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM', 'Florida Panthers': 'FLA',
            'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN', 'Montreal Canadiens': 'MTL',
            'Nashville Predators': 'NSH', 'New Jersey Devils': 'NJD', 'New York Islanders': 'NYI',
            'New York Rangers': 'NYR', 'Ottawa Senators': 'OTT', 'Philadelphia Flyers': 'PHI',
            'Pittsburgh Penguins': 'PIT', 'San Jose Sharks': 'SJS', 'Seattle Kraken': 'SEA',
            'St Louis Blues': 'STL', 'St. Louis Blues': 'STL', 'Tampa Bay Lightning': 'TBL',
            'Toronto Maple Leafs': 'TOR', 'Utah Hockey Club': 'UTA', 'Vancouver Canucks': 'VAN',
            'Vegas Golden Knights': 'VGK', 'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
        }
        
        for game in data:
            home_full = game.get('home_team', '')
            away_full = game.get('away_team', '')
            
            home = name_to_abbrev.get(home_full, home_full)
            away = name_to_abbrev.get(away_full, away_full)
            
            key = f"{away}@{home}"
            odds_map[key] = {'home_team': home, 'away_team': away}
            
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
                break
        
        return odds_map
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return {}


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def main():
    print("=" * 70)
    print(f"NHL MODEL V3 - CURRENT STATE PREDICTIONS - {date.today()}")
    print("=" * 70)
    
    # Load all current data
    print("\n📊 Loading current data...")
    power_ratings = load_power_ratings()
    print(f"  Power ratings: {len(power_ratings)} teams")
    
    injuries = load_injuries()
    injury_count = sum(len(v) for v in injuries.values())
    print(f"  Injuries: {injury_count} players across {len(injuries)} teams")
    
    goalie_stats = load_goalie_stats()
    print(f"  Goalie stats: {len(goalie_stats)} entries")
    
    # Get today's games
    games = get_todays_games()
    print(f"\n🏒 Found {len(games)} games today")
    
    # Get current odds
    odds_map = get_current_odds()
    print(f"💰 Odds loaded for {len(odds_map)} games")
    
    # Generate predictions
    print("\n" + "-" * 70)
    print("PREDICTIONS")
    print("-" * 70)
    
    results = []
    
    for game in games:
        home = game['home_team']
        away = game['away_team']
        
        pred = predict_game(home, away, power_ratings, goalie_stats, injuries)
        
        # Get odds
        game_key = f"{away}@{home}"
        odds_data = odds_map.get(game_key, {})
        home_odds = odds_data.get('home_odds')
        away_odds = odds_data.get('away_odds')
        
        # Calculate edge
        if home_odds:
            market_home = american_to_prob(home_odds)
            home_edge = (pred['home_prob'] - market_home) * 100
        else:
            market_home = None
            home_edge = None
        
        if away_odds:
            market_away = american_to_prob(away_odds)
            away_edge = (pred['away_prob'] - market_away) * 100
        else:
            market_away = None
            away_edge = None
        
        # Print prediction
        print(f"\n{away} @ {home}")
        print(f"  Power: {away} ({pred['away_power']:.1f}) vs {home} ({pred['home_power']:.1f})")
        print(f"  Goalies: {pred['away_goalie']} ({pred['away_goalie_rating']:.0f}) vs {pred['home_goalie']} ({pred['home_goalie_rating']:.0f})")
        
        if pred['home_backup']:
            print(f"  ⚠️  {home} using BACKUP goalie")
        if pred['away_backup']:
            print(f"  ⚠️  {away} using BACKUP goalie")
        
        print(f"  Model: {home} {pred['home_prob']:.1%} | {away} {pred['away_prob']:.1%}")
        
        if home_odds and away_odds:
            print(f"  Odds:  {home} ({home_odds:+d}) | {away} ({away_odds:+d})")
            print(f"  Market: {home} {market_home:.1%} | {away} {market_away:.1%}")
            if home_edge and away_edge:
                print(f"  Edge:  {home} {home_edge:+.1f}% | {away} {away_edge:+.1f}%")
                
                # Flag value plays
                if home_edge > 3:
                    print(f"  >>> VALUE: {home} {home_odds:+d} ({home_edge:+.1f}% edge)")
                elif away_edge > 3:
                    print(f"  >>> VALUE: {away} {away_odds:+d} ({away_edge:+.1f}% edge)")
        
        results.append({
            'game': f"{away} @ {home}",
            'prediction': pred,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'home_edge': home_edge,
            'away_edge': away_edge,
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("VALUE PLAYS (3%+ edge)")
    print("=" * 70)
    
    value_plays = []
    for r in results:
        if r['home_edge'] and r['home_edge'] > 3:
            value_plays.append({
                'game': r['game'],
                'pick': r['prediction']['home_team'],
                'prob': r['prediction']['home_prob'],
                'odds': r['home_odds'],
                'edge': r['home_edge'],
            })
        elif r['away_edge'] and r['away_edge'] > 3:
            value_plays.append({
                'game': r['game'],
                'pick': r['prediction']['away_team'],
                'prob': r['prediction']['away_prob'],
                'odds': r['away_odds'],
                'edge': r['away_edge'],
            })
    
    if value_plays:
        for vp in sorted(value_plays, key=lambda x: -x['edge']):
            print(f"{vp['game']}: {vp['pick']} ({vp['odds']:+d})")
            print(f"  Model: {vp['prob']:.1%} | Edge: {vp['edge']:+.1f}%")
    else:
        print("No value plays found with 3%+ edge")
    
    return results


if __name__ == '__main__':
    main()

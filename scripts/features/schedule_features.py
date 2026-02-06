#!/usr/bin/env python3
"""
Schedule Features - Rest, travel, and situational factors.

Key features:
1. Rest days (days since last game)
2. Back-to-back indicator
3. Home/road trip game number
4. Travel distance (arena to arena)
5. Time zone changes
6. Games in last 7 days (fatigue)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from math import radians, sin, cos, sqrt, atan2

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

# NHL Arena coordinates (lat, lon)
ARENA_COORDS = {
    'ANA': (33.8078, -117.8765),  # Anaheim
    'ARI': (33.5319, -112.2610),  # Arizona (now Utah)
    'BOS': (42.3662, -71.0621),   # Boston
    'BUF': (42.8750, -78.8764),   # Buffalo
    'CGY': (51.0375, -114.0519),  # Calgary
    'CAR': (35.8033, -78.7220),   # Carolina
    'CHI': (41.8807, -87.6742),   # Chicago
    'COL': (39.7487, -105.0077),  # Colorado
    'CBJ': (39.9693, -83.0060),   # Columbus
    'DAL': (32.7905, -96.8103),   # Dallas
    'DET': (42.3411, -83.0550),   # Detroit
    'EDM': (53.5469, -113.4979),  # Edmonton
    'FLA': (26.1584, -80.3256),   # Florida
    'LAK': (34.0430, -118.2673),  # Los Angeles
    'MIN': (44.9448, -93.1010),   # Minnesota
    'MTL': (45.4960, -73.5693),   # Montreal
    'NSH': (36.1592, -86.7785),   # Nashville
    'NJD': (40.7335, -74.1712),   # New Jersey
    'NYI': (40.6826, -73.9754),   # NY Islanders
    'NYR': (40.7505, -73.9934),   # NY Rangers
    'OTT': (45.2969, -75.9269),   # Ottawa
    'PHI': (39.9012, -75.1720),   # Philadelphia
    'PIT': (40.4396, -79.9892),   # Pittsburgh
    'SJS': (37.3327, -121.9012),  # San Jose
    'SEA': (47.6221, -122.3540),  # Seattle
    'STL': (38.6268, -90.2025),   # St. Louis
    'TBL': (27.9426, -82.4519),   # Tampa Bay
    'TOR': (43.6435, -79.3791),   # Toronto
    'UTA': (40.7683, -111.9011),  # Utah (formerly Arizona)
    'VAN': (49.2778, -123.1088),  # Vancouver
    'VGK': (36.1029, -115.1783),  # Vegas
    'WPG': (49.8928, -97.1436),   # Winnipeg
    'WSH': (38.8981, -77.0209),   # Washington
    # Old codes
    'L.A': (34.0430, -118.2673),
    'N.J': (40.7335, -74.1712),
    'S.J': (37.3327, -121.9012),
    'T.B': (27.9426, -82.4519),
}

# Time zones (hours from UTC)
TIMEZONE = {
    'ANA': -8, 'ARI': -7, 'BOS': -5, 'BUF': -5, 'CGY': -7, 'CAR': -5,
    'CHI': -6, 'COL': -7, 'CBJ': -5, 'DAL': -6, 'DET': -5, 'EDM': -7,
    'FLA': -5, 'LAK': -8, 'MIN': -6, 'MTL': -5, 'NSH': -6, 'NJD': -5,
    'NYI': -5, 'NYR': -5, 'OTT': -5, 'PHI': -5, 'PIT': -5, 'SJS': -8,
    'SEA': -8, 'STL': -6, 'TBL': -5, 'TOR': -5, 'UTA': -7, 'VAN': -8,
    'VGK': -8, 'WPG': -6, 'WSH': -5,
    'L.A': -8, 'N.J': -5, 'S.J': -8, 'T.B': -5,
}


def haversine_distance(coord1, coord2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def calculate_travel_distance(from_team, to_team):
    """Calculate travel distance between teams."""
    coord1 = ARENA_COORDS.get(from_team)
    coord2 = ARENA_COORDS.get(to_team)
    
    if coord1 and coord2:
        return haversine_distance(coord1, coord2)
    return 0


def calculate_timezone_change(from_team, to_team):
    """Calculate timezone change (hours)."""
    tz1 = TIMEZONE.get(from_team, -5)
    tz2 = TIMEZONE.get(to_team, -5)
    return abs(tz2 - tz1)


def build_schedule_features(games_df):
    """
    Add schedule features to games dataframe.
    
    Expects columns: gameDate, h_team, a_team
    """
    if games_df.empty:
        return games_df
    
    df = games_df.copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    df = df.sort_values('gameDate')
    
    # Track last game date for each team
    team_last_game = {}
    team_last_location = {}
    team_recent_games = {}  # Track games in last 7 days
    
    features = []
    
    for idx, row in df.iterrows():
        h_team = row['h_team']
        a_team = row['a_team']
        game_date = row['gameDate']
        
        # Initialize tracking
        for team in [h_team, a_team]:
            if team not in team_recent_games:
                team_recent_games[team] = []
        
        # Calculate rest days
        h_rest = (game_date - team_last_game.get(h_team, game_date - timedelta(days=3))).days
        a_rest = (game_date - team_last_game.get(a_team, game_date - timedelta(days=3))).days
        
        # Back-to-back
        h_b2b = 1 if h_rest <= 1 else 0
        a_b2b = 1 if a_rest <= 1 else 0
        
        # Travel distance (away team traveled to home arena)
        a_travel = calculate_travel_distance(
            team_last_location.get(a_team, a_team),  # From last location
            h_team  # To home arena
        )
        
        # Timezone change
        a_tz_change = calculate_timezone_change(
            team_last_location.get(a_team, a_team),
            h_team
        )
        
        # Games in last 7 days
        cutoff = game_date - timedelta(days=7)
        h_games_7d = len([d for d in team_recent_games.get(h_team, []) if d > cutoff])
        a_games_7d = len([d for d in team_recent_games.get(a_team, []) if d > cutoff])
        
        features.append({
            'gameId': row.get('gameId', idx),
            'h_rest_days': h_rest,
            'a_rest_days': a_rest,
            'h_b2b': h_b2b,
            'a_b2b': a_b2b,
            'rest_diff': h_rest - a_rest,  # Positive = home more rested
            'a_travel_km': a_travel,
            'a_tz_change': a_tz_change,
            'h_games_7d': h_games_7d,
            'a_games_7d': a_games_7d,
            'fatigue_diff': a_games_7d - h_games_7d,  # Positive = away more fatigued
        })
        
        # Update tracking
        team_last_game[h_team] = game_date
        team_last_game[a_team] = game_date
        team_last_location[h_team] = h_team  # Home team stays home
        team_last_location[a_team] = h_team  # Away team is now at home arena
        team_recent_games[h_team].append(game_date)
        team_recent_games[a_team].append(game_date)
    
    features_df = pd.DataFrame(features)
    
    # Drop columns that already exist in df
    existing_cols = set(df.columns)
    new_cols = [c for c in features_df.columns if c not in existing_cols and c != 'gameId']
    features_df = features_df[new_cols]
    
    # Merge back - just concat since we processed in order
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    return result


def analyze_schedule_impact(df):
    """Analyze how schedule factors correlate with home wins."""
    if 'home_win' not in df.columns:
        print("No home_win column - skipping analysis")
        return
    
    print("\n📊 Schedule Factor Impact on Home Win Rate:")
    print("-" * 50)
    
    # Check required columns exist
    required = ['h_b2b', 'a_b2b', 'rest_diff', 'a_travel_km']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(f"Available: {[c for c in df.columns if 'rest' in c.lower() or 'b2b' in c.lower() or 'travel' in c.lower()]}")
        return
    
    # Back-to-back analysis
    home_b2b = df[df['h_b2b'] == 1]['home_win'].mean()
    away_b2b = df[df['a_b2b'] == 1]['home_win'].mean()
    neither_b2b = df[(df['h_b2b'] == 0) & (df['a_b2b'] == 0)]['home_win'].mean()
    
    print(f"Home B2B:    {home_b2b:.1%} ({len(df[df['h_b2b']==1])} games)")
    print(f"Away B2B:    {away_b2b:.1%} ({len(df[df['a_b2b']==1])} games)")
    print(f"Neither B2B: {neither_b2b:.1%} ({len(df[(df['h_b2b']==0)&(df['a_b2b']==0)])} games)")
    
    # Rest advantage
    print("\nRest advantage:")
    rest_bins = [(-10, -1), (0, 0), (1, 2), (3, 10)]
    for low, high in rest_bins:
        mask = (df['rest_diff'] >= low) & (df['rest_diff'] <= high)
        subset = df.loc[mask]
        if len(subset) > 50:
            print(f"  Rest diff {low:+d} to {high:+d}: {subset['home_win'].mean():.1%} ({len(subset)} games)")
    
    # Travel impact (binned)
    print("\nAway team travel:")
    travel_bins = [(0, 500), (500, 1500), (1500, 3000), (3000, 10000)]
    for low, high in travel_bins:
        mask = (df['a_travel_km'] >= low) & (df['a_travel_km'] < high)
        subset = df.loc[mask]
        if len(subset) > 50:
            print(f"  {low}-{high}km: {subset['home_win'].mean():.1%} ({len(subset)} games)")


def main():
    print("=" * 60)
    print("SCHEDULE FEATURES")
    print("=" * 60)
    
    # Load matchups data
    matchups_path = DATA_DIR / 'processed' / 'matchups_featured.csv'
    if not matchups_path.exists():
        print(f"❌ Matchups file not found: {matchups_path}")
        return
    
    df = pd.read_csv(matchups_path)
    print(f"Loaded {len(df)} matchups")
    
    # Build schedule features
    print("\n🏗️  Building schedule features...")
    df_with_schedule = build_schedule_features(df)
    
    # Analyze impact
    analyze_schedule_impact(df_with_schedule)
    
    # Save enhanced dataset
    output_path = DATA_DIR / 'processed' / 'matchups_with_schedule.csv'
    df_with_schedule.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    
    # Sample output
    print("\nSample features:")
    print(df_with_schedule[['gameDate', 'h_team', 'a_team', 'h_rest_days', 'a_rest_days', 
                            'h_b2b', 'a_b2b', 'a_travel_km']].tail(10).to_string())
    
    return df_with_schedule


if __name__ == '__main__':
    main()

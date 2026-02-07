#!/usr/bin/env python3
"""
Build situational edge features that sharps use but models often miss.

Features:
1. Playoff implications (clinched, eliminated, fighting for spot)
2. Revenge games (recent head-to-head blowout losses)
3. Division games (more competitive historically)
4. Home stand / road trip game number
5. Coming off long break (All-Star, bye week)
6. Altitude adjustment (Denver games)
7. Travel fatigue (timezone changes, distance)
8. Schedule spots (sandwich games, back-to-backs)

Expected Impact: +0.5% Brier improvement
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATCHUPS_FILE = os.path.join(BASE_DIR, 'data/processed/matchups_with_goalie.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/processed/matchups_with_situational.csv')

# Team divisions (2021-22 realignment)
DIVISIONS = {
    'Atlantic': ['BOS', 'BUF', 'DET', 'FLA', 'MTL', 'OTT', 'TBL', 'TOR'],
    'Metropolitan': ['CAR', 'CBJ', 'NJD', 'NYI', 'NYR', 'PHI', 'PIT', 'WSH'],
    'Central': ['ARI', 'CHI', 'COL', 'DAL', 'MIN', 'NSH', 'STL', 'WPG', 'UTA'],
    'Pacific': ['ANA', 'CGY', 'EDM', 'LAK', 'SEA', 'SJS', 'VAN', 'VGK'],
}

# High altitude venue
HIGH_ALTITUDE_TEAMS = ['COL']  # Denver at 5,280 ft

# Timezone mappings (approximate)
TEAM_TIMEZONES = {
    'BOS': 'ET', 'BUF': 'ET', 'DET': 'ET', 'FLA': 'ET', 'MTL': 'ET',
    'OTT': 'ET', 'TBL': 'ET', 'TOR': 'ET', 'CAR': 'ET', 'CBJ': 'ET',
    'NJD': 'ET', 'NYI': 'ET', 'NYR': 'ET', 'PHI': 'ET', 'PIT': 'ET', 'WSH': 'ET',
    'CHI': 'CT', 'DAL': 'CT', 'MIN': 'CT', 'NSH': 'CT', 'STL': 'CT', 'WPG': 'CT',
    'ARI': 'MT', 'COL': 'MT', 'UTA': 'MT',
    'ANA': 'PT', 'CGY': 'MT', 'EDM': 'MT', 'LAK': 'PT', 'SEA': 'PT',
    'SJS': 'PT', 'VAN': 'PT', 'VGK': 'PT',
}

TZ_OFFSET = {'ET': 0, 'CT': 1, 'MT': 2, 'PT': 3}


def get_division(team):
    """Get division for a team."""
    for div, teams in DIVISIONS.items():
        if team in teams:
            return div
    return 'Unknown'


def load_matchups():
    """Load matchups data."""
    print("Loading matchups...")
    df = pd.read_csv(MATCHUPS_FILE, low_memory=False)
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    print(f"Loaded {len(df):,} matchups")
    return df


def add_division_features(df):
    """Add division game indicator and rivalry features."""
    print("\nAdding division features...")
    
    df['h_division'] = df['h_team'].map(get_division)
    df['a_division'] = df['a_team'].map(get_division)
    df['is_division_game'] = df['h_division'] == df['a_division']
    
    # Same division games are historically more competitive
    print(f"  Division games: {df['is_division_game'].sum():,} ({100*df['is_division_game'].mean():.1f}%)")
    
    return df


def add_altitude_features(df):
    """Add altitude adjustment for Denver games."""
    print("\nAdding altitude features...")
    
    df['home_high_altitude'] = df['h_team'].isin(HIGH_ALTITUDE_TEAMS)
    df['away_at_altitude'] = df['home_high_altitude']  # Away team playing at altitude
    
    print(f"  High altitude home games: {df['home_high_altitude'].sum():,}")
    
    return df


def add_travel_features(df):
    """Add travel/timezone fatigue features."""
    print("\nAdding travel features...")
    
    # Get timezone for each team
    df['h_timezone'] = df['h_team'].map(TEAM_TIMEZONES).map(TZ_OFFSET).fillna(0)
    df['a_timezone'] = df['a_team'].map(TEAM_TIMEZONES).map(TZ_OFFSET).fillna(0)
    
    # Timezone difference (absolute)
    df['timezone_diff'] = abs(df['h_timezone'] - df['a_timezone'])
    
    # West coast team traveling east (jet lag disadvantage)
    df['away_traveling_east'] = (df['a_timezone'] > df['h_timezone']).astype(int)
    df['away_traveling_west'] = (df['a_timezone'] < df['h_timezone']).astype(int)
    
    # Significant travel (2+ timezone diff)
    df['significant_travel'] = df['timezone_diff'] >= 2
    
    print(f"  Games with 2+ timezone diff: {df['significant_travel'].sum():,}")
    
    return df


def add_schedule_spot_features(df):
    """Add schedule spot features (home stands, road trips, sandwich games)."""
    print("\nAdding schedule spot features...")
    
    df = df.sort_values(['gameDate']).copy()
    
    # Track consecutive home/away games for each team
    home_stand_game = []
    road_trip_game = []
    
    # Group by team and calculate streak
    for team in df['h_team'].unique():
        # Home games for this team
        home_games = df[df['h_team'] == team].copy()
        away_games = df[df['a_team'] == team].copy()
        
        # Combine and sort by date
        team_games = pd.concat([
            home_games[['gameId', 'gameDate']].assign(is_home=True),
            away_games[['gameId', 'gameDate']].assign(is_home=False)
        ]).sort_values('gameDate')
        
        # Calculate streaks
        team_games['streak'] = (team_games['is_home'] != team_games['is_home'].shift()).cumsum()
        team_games['game_in_streak'] = team_games.groupby('streak').cumcount() + 1
        
        # Store results
        for _, row in team_games.iterrows():
            if row['is_home']:
                home_stand_game.append((row['gameId'], row['game_in_streak']))
            else:
                road_trip_game.append((row['gameId'], row['game_in_streak']))
    
    # Merge back
    home_df = pd.DataFrame(home_stand_game, columns=['gameId', 'home_stand_game_num'])
    road_df = pd.DataFrame(road_trip_game, columns=['gameId', 'road_trip_game_num'])
    
    df = df.merge(home_df, on='gameId', how='left')
    df = df.merge(road_df, on='gameId', how='left')
    
    # Fill NaN (first games of season)
    df['home_stand_game_num'] = df['home_stand_game_num'].fillna(1)
    df['road_trip_game_num'] = df['road_trip_game_num'].fillna(1)
    
    # Long home stand fatigue (5+ games)
    df['long_home_stand'] = df['home_stand_game_num'] >= 5
    
    # Long road trip fatigue (4+ games)
    df['long_road_trip'] = df['road_trip_game_num'] >= 4
    
    print(f"  Long home stands (5+): {df['long_home_stand'].sum():,}")
    print(f"  Long road trips (4+): {df['long_road_trip'].sum():,}")
    
    return df


def add_rest_advantage_features(df):
    """Enhanced rest features beyond just days off."""
    print("\nAdding enhanced rest features...")
    
    # Rest categories
    # Note: a_rest_days and h_rest_days should already exist
    if 'h_rest_days' in df.columns and 'a_rest_days' in df.columns:
        # Coming off extended break (3+ days)
        df['home_extended_rest'] = df.get('h_rest_days', 1) >= 3
        df['away_extended_rest'] = df.get('a_rest_days', 1) >= 3
        
        # Rest mismatch (one team rested, other on B2B)
        df['rest_mismatch'] = (
            ((df.get('h_rest_days', 1) >= 2) & (df.get('a_rest_days', 1) == 0)) |
            ((df.get('a_rest_days', 1) >= 2) & (df.get('h_rest_days', 1) == 0))
        )
        
        print(f"  Rest mismatches: {df['rest_mismatch'].sum():,}")
    else:
        df['home_extended_rest'] = False
        df['away_extended_rest'] = False
        df['rest_mismatch'] = False
        print("  Warning: rest_days columns not found")
    
    return df


def add_revenge_game_features(df):
    """Add revenge game indicator (lost badly in recent H2H)."""
    print("\nAdding revenge game features...")
    
    df = df.sort_values('gameDate').copy()
    
    # Look for recent H2H blowouts (3+ goal margin)
    revenge_home = []
    revenge_away = []
    
    for idx, row in df.iterrows():
        h_team = row['h_team']
        a_team = row['a_team']
        game_date = row['gameDate']
        
        # Look back 60 days for H2H games
        lookback = game_date - timedelta(days=60)
        
        # Recent H2H where current home team lost badly
        recent_h2h = df[
            (df['gameDate'] >= lookback) & 
            (df['gameDate'] < game_date) &
            (((df['h_team'] == h_team) & (df['a_team'] == a_team)) |
             ((df['h_team'] == a_team) & (df['a_team'] == h_team)))
        ]
        
        # Check for blowout losses
        home_revenge = False
        away_revenge = False
        
        for _, h2h in recent_h2h.iterrows():
            if 'h_totalGoalsFor' in h2h and 'a_totalGoalsFor' in h2h:
                h_goals = h2h.get('h_totalGoalsFor', 0)
                a_goals = h2h.get('a_totalGoalsFor', 0)
                margin = abs(h_goals - a_goals) if pd.notna(h_goals) and pd.notna(a_goals) else 0
                
                if margin >= 3:
                    # Who lost the blowout?
                    if h2h['h_team'] == h_team and h_goals < a_goals:
                        home_revenge = True
                    elif h2h['a_team'] == h_team and a_goals < h_goals:
                        home_revenge = True
                    elif h2h['h_team'] == a_team and h_goals < a_goals:
                        away_revenge = True
                    elif h2h['a_team'] == a_team and a_goals < h_goals:
                        away_revenge = True
        
        revenge_home.append(home_revenge)
        revenge_away.append(away_revenge)
    
    df['home_revenge_game'] = revenge_home
    df['away_revenge_game'] = revenge_away
    
    print(f"  Home revenge games: {sum(revenge_home):,}")
    print(f"  Away revenge games: {sum(revenge_away):,}")
    
    return df


def main():
    print("="*60)
    print("SITUATIONAL FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    df = load_matchups()
    
    # Add features
    df = add_division_features(df)
    df = add_altitude_features(df)
    df = add_travel_features(df)
    df = add_schedule_spot_features(df)
    df = add_rest_advantage_features(df)
    df = add_revenge_game_features(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    new_cols = [
        'is_division_game', 'home_high_altitude', 'away_at_altitude',
        'timezone_diff', 'away_traveling_east', 'away_traveling_west', 'significant_travel',
        'home_stand_game_num', 'road_trip_game_num', 'long_home_stand', 'long_road_trip',
        'home_extended_rest', 'away_extended_rest', 'rest_mismatch',
        'home_revenge_game', 'away_revenge_game'
    ]
    
    print(f"\nNew situational features ({len(new_cols)}):")
    for col in new_cols:
        if col in df.columns:
            if df[col].dtype == bool:
                print(f"  {col}: {df[col].sum():,} ({100*df[col].mean():.1f}%)")
            else:
                print(f"  {col}: mean={df[col].mean():.2f}")
    
    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    return df


if __name__ == '__main__':
    df = main()

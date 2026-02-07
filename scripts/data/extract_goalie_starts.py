#!/usr/bin/env python3
"""
Extract game-level goalie starts from MoneyPuck shots data.

For each game:
1. Identify which goalie(s) faced shots for each team
2. Determine the "starting goalie" (faced first 5+ shots in periods 1-2)
3. Flag backup starts and relief appearances
4. Calculate goalie rest days

Output: data/goalie_starts/game_goalie_starts.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHOTS_FILE = os.path.join(BASE_DIR, 'data/raw/shots_2018-2024.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data/goalie_starts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_shots_data():
    """Load shots data with relevant columns only."""
    print("Loading shots data...")
    cols = [
        'game_id', 'season', 'homeTeamCode', 'awayTeamCode', 
        'period', 'time', 'team', 'event',
        'goalieIdForShot', 'goalieNameForShot',
        'isHomeTeam', 'isPlayoffGame'
    ]
    
    df = pd.read_csv(SHOTS_FILE, usecols=cols, dtype={
        'game_id': int,
        'season': int,
        'goalieIdForShot': float,  # Can be NaN
        'period': int,
        'time': int,
        'isHomeTeam': float,
        'isPlayoffGame': int
    })
    
    # Filter to regular season only
    df = df[df['isPlayoffGame'] == 0].copy()
    
    # Construct full game_id to match moneypuck format: YYYYGGNNNN
    # where YYYY=season, GG=02 (regular season), NNNN=game number
    df['full_game_id'] = df['season'] * 1000000 + df['game_id']
    
    print(f"Loaded {len(df):,} shots from {df['game_id'].nunique():,} games")
    return df


def get_game_dates(shots_df):
    """
    Extract game dates from moneypuck_all_games.csv.
    Match on full game_id (YYYYGGNNNN format).
    """
    games_file = os.path.join(BASE_DIR, 'data/raw/moneypuck_all_games.csv')
    
    print("Loading game dates from moneypuck_all_games.csv...")
    games_df = pd.read_csv(games_file, usecols=['gameId', 'gameDate', 'season', 'home_or_away', 'team'])
    
    # Get unique games with dates (one row per game)
    games_df = games_df[games_df['home_or_away'] == 'HOME'][['gameId', 'gameDate', 'season', 'team']].copy()
    games_df.columns = ['game_id', 'game_date', 'season', 'home_team']
    games_df['game_date'] = pd.to_datetime(games_df['game_date'].astype(str), format='%Y%m%d')
    games_df = games_df.drop_duplicates(subset=['game_id'])
    
    # Filter to only games we have shots for
    shots_game_ids = shots_df['full_game_id'].unique()
    games_df = games_df[games_df['game_id'].isin(shots_game_ids)]
    
    print(f"Found dates for {len(games_df):,} games")
    return games_df


def extract_goalie_starts(shots_df, games_df):
    """
    For each game, identify the starting goalie for each team.
    Starting goalie = goalie who faced majority of shots in periods 1-2 (before pulls/injuries)
    """
    print("\nExtracting goalie starts...")
    
    results = []
    
    # Group by full_game_id (includes season)
    for full_game_id, game_shots in shots_df.groupby('full_game_id'):
        home_team = game_shots['homeTeamCode'].iloc[0]
        away_team = game_shots['awayTeamCode'].iloc[0]
        season = game_shots['season'].iloc[0]
        
        # Get shots by period (1-2 for identifying starter)
        early_shots = game_shots[game_shots['period'] <= 2]
        all_shots = game_shots
        
        # Home team's goalie faces shots when isHomeTeam == 0 (away team shooting)
        home_goalie_shots = all_shots[all_shots['isHomeTeam'] == 0].copy()
        away_goalie_shots = all_shots[all_shots['isHomeTeam'] == 1].copy()
        
        home_goalie_early = early_shots[early_shots['isHomeTeam'] == 0].copy()
        away_goalie_early = early_shots[early_shots['isHomeTeam'] == 1].copy()
        
        # Identify home starting goalie (most shots faced in P1-P2)
        if len(home_goalie_early) > 0:
            home_starter_counts = home_goalie_early.groupby(['goalieIdForShot', 'goalieNameForShot']).size()
            if len(home_starter_counts) > 0:
                home_starter_idx = home_starter_counts.idxmax()
                home_starter_id = home_starter_idx[0]
                home_starter_name = home_starter_idx[1]
                home_starter_early_shots = home_starter_counts.max()
            else:
                home_starter_id, home_starter_name, home_starter_early_shots = np.nan, None, 0
        else:
            home_starter_id, home_starter_name, home_starter_early_shots = np.nan, None, 0
        
        # Identify away starting goalie
        if len(away_goalie_early) > 0:
            away_starter_counts = away_goalie_early.groupby(['goalieIdForShot', 'goalieNameForShot']).size()
            if len(away_starter_counts) > 0:
                away_starter_idx = away_starter_counts.idxmax()
                away_starter_id = away_starter_idx[0]
                away_starter_name = away_starter_idx[1]
                away_starter_early_shots = away_starter_counts.max()
            else:
                away_starter_id, away_starter_name, away_starter_early_shots = np.nan, None, 0
        else:
            away_starter_id, away_starter_name, away_starter_early_shots = np.nan, None, 0
        
        # Check for relief appearances (multiple goalies)
        home_goalies = home_goalie_shots['goalieIdForShot'].dropna().unique()
        away_goalies = away_goalie_shots['goalieIdForShot'].dropna().unique()
        
        home_relief = len(home_goalies) > 1
        away_relief = len(away_goalies) > 1
        
        # Total shots faced by starter (all periods)
        home_starter_total_shots = len(home_goalie_shots[home_goalie_shots['goalieIdForShot'] == home_starter_id])
        away_starter_total_shots = len(away_goalie_shots[away_goalie_shots['goalieIdForShot'] == away_starter_id])
        
        results.append({
            'game_id': full_game_id,
            'season': season,
            'home_team': home_team,
            'away_team': away_team,
            'home_goalie_id': home_starter_id,
            'home_goalie_name': home_starter_name,
            'home_goalie_shots_early': home_starter_early_shots,
            'home_goalie_shots_total': home_starter_total_shots,
            'home_relief_appearance': home_relief,
            'away_goalie_id': away_starter_id,
            'away_goalie_name': away_starter_name,
            'away_goalie_shots_early': away_starter_early_shots,
            'away_goalie_shots_total': away_starter_total_shots,
            'away_relief_appearance': away_relief,
        })
        
        if len(results) % 1000 == 0:
            print(f"  Processed {len(results):,} games...")
    
    df = pd.DataFrame(results)
    
    # Merge with game dates
    df = df.merge(games_df[['game_id', 'game_date']], on='game_id', how='left')
    
    return df


def calculate_goalie_rest(df):
    """
    Calculate days since last start for each goalie.
    """
    print("\nCalculating goalie rest days...")
    
    # Sort by date
    df = df.sort_values('game_date').copy()
    
    # Track last game for each goalie
    goalie_last_game = {}
    
    home_rest = []
    away_rest = []
    
    for _, row in df.iterrows():
        game_date = row['game_date']
        
        # Home goalie rest
        h_goalie = row['home_goalie_id']
        if pd.notna(h_goalie) and h_goalie in goalie_last_game:
            h_rest = (game_date - goalie_last_game[h_goalie]).days
        else:
            h_rest = np.nan  # First game or unknown
        home_rest.append(h_rest)
        if pd.notna(h_goalie):
            goalie_last_game[h_goalie] = game_date
        
        # Away goalie rest
        a_goalie = row['away_goalie_id']
        if pd.notna(a_goalie) and a_goalie in goalie_last_game:
            a_rest = (game_date - goalie_last_game[a_goalie]).days
        else:
            a_rest = np.nan
        away_rest.append(a_rest)
        if pd.notna(a_goalie):
            goalie_last_game[a_goalie] = game_date
    
    df['home_goalie_rest_days'] = home_rest
    df['away_goalie_rest_days'] = away_rest
    
    # Flag back-to-back starts (0 rest days = same day is impossible, 1 day = B2B)
    df['home_goalie_b2b'] = df['home_goalie_rest_days'] == 1
    df['away_goalie_b2b'] = df['away_goalie_rest_days'] == 1
    
    return df


def identify_backup_starts(df):
    """
    Identify backup goalie starts by comparing to team's primary goalie.
    Backup start = goalie started less than 40% of team's games in last 30 days
    """
    print("\nIdentifying backup starts...")
    
    df = df.sort_values('game_date').copy()
    
    # Calculate rolling goalie starts per team
    home_backup = []
    away_backup = []
    
    for idx, row in df.iterrows():
        game_date = row['game_date']
        
        # Look back 30 days
        window_start = game_date - pd.Timedelta(days=30)
        
        # Home team's recent games
        team = row['home_team']
        goalie = row['home_goalie_id']
        
        recent_home = df[
            (df['home_team'] == team) & 
            (df['game_date'] >= window_start) & 
            (df['game_date'] < game_date)
        ]
        
        if len(recent_home) >= 5 and pd.notna(goalie):
            goalie_starts = (recent_home['home_goalie_id'] == goalie).sum()
            start_pct = goalie_starts / len(recent_home)
            home_backup.append(start_pct < 0.4)
        else:
            home_backup.append(False)  # Not enough data to determine
        
        # Away team's recent games
        team = row['away_team']
        goalie = row['away_goalie_id']
        
        recent_away = df[
            (df['away_team'] == team) & 
            (df['game_date'] >= window_start) & 
            (df['game_date'] < game_date)
        ]
        
        if len(recent_away) >= 5 and pd.notna(goalie):
            goalie_starts = (recent_away['away_goalie_id'] == goalie).sum()
            start_pct = goalie_starts / len(recent_away)
            away_backup.append(start_pct < 0.4)
        else:
            away_backup.append(False)
    
    df['home_backup_start'] = home_backup
    df['away_backup_start'] = away_backup
    
    return df


def main():
    print("="*60)
    print("GOALIE STARTS EXTRACTION")
    print("="*60)
    
    # Load data
    shots_df = load_shots_data()
    games_df = get_game_dates(shots_df)
    
    # Extract goalie starts
    df = extract_goalie_starts(shots_df, games_df)
    
    # Calculate rest days
    df = calculate_goalie_rest(df)
    
    # Identify backup starts
    df = identify_backup_starts(df)
    
    # Summary stats
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total games: {len(df):,}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"\nHome backup starts: {df['home_backup_start'].sum():,} ({100*df['home_backup_start'].mean():.1f}%)")
    print(f"Away backup starts: {df['away_backup_start'].sum():,} ({100*df['away_backup_start'].mean():.1f}%)")
    print(f"\nHome B2B starts: {df['home_goalie_b2b'].sum():,} ({100*df['home_goalie_b2b'].mean():.1f}%)")
    print(f"Away B2B starts: {df['away_goalie_b2b'].sum():,} ({100*df['away_goalie_b2b'].mean():.1f}%)")
    print(f"\nRelief appearances (home): {df['home_relief_appearance'].sum():,}")
    print(f"Relief appearances (away): {df['away_relief_appearance'].sum():,}")
    
    # Sample
    print("\nSample data:")
    print(df[['game_date', 'home_team', 'away_team', 
              'home_goalie_name', 'home_goalie_rest_days', 'home_backup_start',
              'away_goalie_name', 'away_goalie_rest_days', 'away_backup_start']].tail(10))
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, 'game_goalie_starts.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return df


if __name__ == '__main__':
    df = main()

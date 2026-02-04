#!/usr/bin/env python3
"""
Build training features from raw MoneyPuck game-level data.

Key principle: All features must be calculated BEFORE the game.
No leakage - only use data available at prediction time.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_game_data():
    """Load MoneyPuck game-level team data
    
    Uses 'all' situation for game outcomes (actual winner)
    Uses '5on5' situation for predictive features (team quality)
    """
    path = os.path.join(RAW_DIR, 'moneypuck_all_games.csv')
    df_raw = pd.read_csv(path)
    
    # Parse date
    df_raw['gameDate'] = pd.to_datetime(df_raw['gameDate'].astype(str), format='%Y%m%d')
    
    # Get 'all' situation for actual game outcomes
    df_all = df_raw[df_raw['situation'] == 'all'][['gameId', 'team', 'home_or_away', 'goalsFor', 'goalsAgainst']].copy()
    df_all = df_all.rename(columns={'goalsFor': 'totalGoalsFor', 'goalsAgainst': 'totalGoalsAgainst'})
    
    # Get 5on5 situation for predictive features
    df = df_raw[df_raw['situation'] == '5on5'].copy()
    
    # Merge actual outcomes
    df = df.merge(df_all, on=['gameId', 'team', 'home_or_away'], how='left')
    
    print(f"Loaded {len(df)} game records")
    print(f"Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    
    return df


def compute_rolling_features(df, windows=[10, 20, 40]):
    """
    Compute rolling averages for each team.
    
    CRITICAL: Use shift(1) to prevent leakage - only use data from BEFORE the game.
    """
    
    print("\nComputing rolling features...")
    
    # Sort by team and date
    df = df.sort_values(['team', 'gameDate']).copy()
    
    # Key metrics to roll
    metrics = [
        'xGoalsFor', 'xGoalsAgainst', 'goalsFor', 'goalsAgainst',
        'corsiPercentage', 'fenwickPercentage', 'xGoalsPercentage',
        'shotsOnGoalFor', 'shotsOnGoalAgainst',
        'highDangerShotsFor', 'highDangerShotsAgainst'
    ]
    
    # Compute per-game metrics first
    if 'iceTime' in df.columns:
        df['xGF_per60'] = df['xGoalsFor'] / (df['iceTime'] / 60) * 60
        df['xGA_per60'] = df['xGoalsAgainst'] / (df['iceTime'] / 60) * 60
        metrics.extend(['xGF_per60', 'xGA_per60'])
    
    results = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        
        for window in windows:
            for metric in metrics:
                if metric in team_df.columns:
                    # SHIFT(1) - use only prior games, not current
                    col_name = f'{metric}_L{window}'
                    team_df[col_name] = (
                        team_df[metric]
                        .shift(1)
                        .rolling(window, min_periods=max(3, window//2))
                        .mean()
                    )
        
        # Win rate
        team_df['win'] = (team_df['goalsFor'] > team_df['goalsAgainst']).astype(int)
        for window in windows:
            team_df[f'winPct_L{window}'] = (
                team_df['win']
                .shift(1)
                .rolling(window, min_periods=max(3, window//2))
                .mean()
            )
        
        results.append(team_df)
    
    combined = pd.concat(results, ignore_index=True)
    print(f"  Added {len([c for c in combined.columns if '_L' in c])} rolling features")
    
    return combined


def create_matchups(df):
    """
    Create matchup dataset - one row per game with home and away features.
    """
    
    print("\nCreating matchups...")
    
    # Split into home and away
    home = df[df['home_or_away'] == 'HOME'].copy()
    away = df[df['home_or_away'] == 'AWAY'].copy()
    
    # Rename columns
    home_cols = {c: f'h_{c}' for c in home.columns if c not in ['gameId', 'gameDate', 'season']}
    away_cols = {c: f'a_{c}' for c in away.columns if c not in ['gameId', 'gameDate', 'season']}
    
    home = home.rename(columns=home_cols)
    away = away.rename(columns=away_cols)
    
    # Merge on game ID
    matchups = home.merge(
        away,
        on=['gameId', 'gameDate', 'season'],
        how='inner'
    )
    
    print(f"  Created {len(matchups)} matchups")
    
    # Create target: home win (using total goals, not 5on5)
    matchups['home_win'] = (matchups['h_totalGoalsFor'] > matchups['h_totalGoalsAgainst']).astype(int)
    
    # Create differentials - use L20 and L40 for more stability
    diff_pairs = [
        ('xGoalsPercentage', 'L20'),
        ('xGoalsPercentage', 'L40'),
        ('corsiPercentage', 'L20'),
        ('corsiPercentage', 'L40'),
        ('xGoalsFor', 'L20'),
        ('xGoalsAgainst', 'L20'),
        ('winPct', 'L20'),
        ('winPct', 'L40'),
    ]
    
    for metric, window in diff_pairs:
        h_col = f'h_{metric}_{window}'
        a_col = f'a_{metric}_{window}'
        if h_col in matchups.columns and a_col in matchups.columns:
            matchups[f'{metric}_{window}_diff'] = matchups[h_col] - matchups[a_col]
    
    return matchups


def add_rest_days(df):
    """Add rest days and home ice features"""
    
    print("\nAdding rest days and home ice...")
    
    df = df.sort_values(['h_team', 'gameDate']).copy()
    
    # Calculate rest for home team
    for prefix in ['h', 'a']:
        team_col = f'{prefix}_team'
        df[f'{prefix}_prev_game'] = df.groupby(team_col)['gameDate'].shift(1)
        df[f'{prefix}_rest_days'] = (df['gameDate'] - df[f'{prefix}_prev_game']).dt.days
        df[f'{prefix}_rest_days'] = df[f'{prefix}_rest_days'].clip(1, 7).fillna(3)
    
    df['rest_diff'] = df['h_rest_days'] - df['a_rest_days']
    
    return df


def filter_features(df):
    """Select final feature set for modeling"""
    
    # Core features we want
    feature_patterns = [
        'xGoalsPercentage_L',
        'corsiPercentage_L', 
        'fenwickPercentage_L',
        'xGoalsFor_L',
        'xGoalsAgainst_L',
        'winPct_L',
        'highDangerShotsFor_L',
        'highDangerShotsAgainst_L',
        '_diff',
        '_rest_days',
        'rest_diff',
    ]
    
    # Identify feature columns
    feature_cols = []
    for col in df.columns:
        for pattern in feature_patterns:
            if pattern in col:
                feature_cols.append(col)
                break
    
    # Add identifiers and target
    id_cols = ['gameId', 'gameDate', 'season', 'h_team', 'a_team', 'home_win', 
               'h_totalGoalsFor', 'h_totalGoalsAgainst', 'a_totalGoalsFor', 'a_totalGoalsAgainst']
    
    keep_cols = [c for c in id_cols if c in df.columns] + list(set(feature_cols))
    
    result = df[keep_cols].copy()
    
    print(f"\nFinal dataset: {len(result)} games, {len(feature_cols)} features")
    
    return result


def main():
    print("="*60)
    print("NHL Model - Feature Engineering")
    print("="*60)
    
    # Load data
    df = load_game_data()
    
    # Compute rolling features
    df = compute_rolling_features(df)
    
    # Create matchups
    matchups = create_matchups(df)
    
    # Add rest days
    matchups = add_rest_days(matchups)
    
    # Filter to final features
    final = filter_features(matchups)
    
    # Drop rows with too many NaNs (early season games)
    initial_count = len(final)
    final = final.dropna(thresh=len(final.columns) - 10)
    print(f"Dropped {initial_count - len(final)} rows with missing data")
    
    # Save
    output_path = os.path.join(PROCESSED_DIR, 'matchups_featured.csv')
    final.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total matchups: {len(final)}")
    print(f"Seasons: {sorted(final['season'].unique())}")
    print(f"Date range: {final['gameDate'].min()} to {final['gameDate'].max()}")
    print(f"Home win rate: {final['home_win'].mean():.1%}")
    print(f"Features: {len([c for c in final.columns if c not in ['gameId', 'gameDate', 'season', 'h_team', 'a_team', 'home_win']])}")
    
    return final


if __name__ == '__main__':
    main()

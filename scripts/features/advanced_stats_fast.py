#!/usr/bin/env python3
"""
Fast version: Add missing advanced statistics using vectorized operations.

New features:
1. Shooting % and Save % (rolling L10, L20)
2. Faceoff win % 
3. High danger chances for/against
4. Recent goal differential
5. Penalty differential
6. Physical play (hits, blocks)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'


def load_data():
    """Load MoneyPuck 5on5 game data."""
    path = DATA_DIR / 'raw' / 'moneypuck_all_games.csv'
    
    print("Loading MoneyPuck data...")
    df = pd.read_csv(path)
    
    # Filter to 5on5, team level
    df = df[(df['situation'] == '5on5') & (df['position'] == 'Team Level')].copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'].astype(str), format='%Y%m%d')
    
    print(f"Loaded {len(df)} team-game records")
    return df


def calculate_team_rolling_stats(df):
    """Calculate rolling stats per team efficiently."""
    
    print("Calculating rolling stats...")
    
    df = df.sort_values(['team', 'gameDate']).copy()
    
    # Create derived columns
    df['shooting_pct'] = df['goalsFor'] / df['shotsOnGoalFor'].replace(0, np.nan)
    df['save_pct'] = 1 - (df['goalsAgainst'] / df['shotsOnGoalAgainst'].replace(0, np.nan))
    df['faceoff_pct'] = df['faceOffsWonFor'] / (df['faceOffsWonFor'] + df['faceOffsWonAgainst']).replace(0, np.nan)
    df['goal_diff'] = df['goalsFor'] - df['goalsAgainst']
    df['penalty_diff'] = df['penaltiesAgainst'] - df['penaltiesFor']  # Positive = draw more penalties
    df['hd_chance_diff'] = df['highDangerShotsFor'] - df['highDangerShotsAgainst']
    
    # Rolling windows
    stats_to_roll = ['shooting_pct', 'save_pct', 'faceoff_pct', 'goal_diff', 
                     'hitsFor', 'blockedShotAttemptsFor', 'takeawaysFor', 'giveawaysFor',
                     'highDangerShotsFor', 'highDangerShotsAgainst', 'hd_chance_diff',
                     'penalty_diff']
    
    for stat in stats_to_roll:
        if stat in df.columns:
            for window in [10, 20]:
                col_name = f'{stat}_L{window}'
                df[col_name] = df.groupby('team')[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=3).mean()
                )
    
    return df


def merge_with_matchups(team_stats):
    """Merge team stats with matchups."""
    
    print("Merging with matchups...")
    
    matchups = pd.read_csv(DATA_DIR / 'processed' / 'matchups_with_situational.csv', low_memory=False)
    matchups['gameDate'] = pd.to_datetime(matchups['gameDate'])
    
    # Select rolling columns
    rolling_cols = [c for c in team_stats.columns if '_L10' in c or '_L20' in c]
    
    # Prepare for merge
    home_stats = team_stats[['gameId', 'team'] + rolling_cols].copy()
    home_stats.columns = ['gameId', 'h_team_mp'] + ['h_' + c for c in rolling_cols]
    
    away_stats = team_stats[['gameId', 'team'] + rolling_cols].copy()
    away_stats.columns = ['gameId', 'a_team_mp'] + ['a_' + c for c in rolling_cols]
    
    # For home stats, we need where home_or_away == 'HOME'
    home_mask = team_stats['home_or_away'] == 'HOME'
    away_mask = team_stats['home_or_away'] == 'AWAY'
    
    home_data = team_stats[home_mask][['gameId'] + rolling_cols].copy()
    home_data.columns = ['gameId'] + ['h_' + c for c in rolling_cols]
    
    away_data = team_stats[away_mask][['gameId'] + rolling_cols].copy()
    away_data.columns = ['gameId'] + ['a_' + c for c in rolling_cols]
    
    # Merge
    merged = matchups.merge(home_data, on='gameId', how='left')
    merged = merged.merge(away_data, on='gameId', how='left')
    
    # Calculate differentials
    for stat in ['shooting_pct', 'save_pct', 'faceoff_pct', 'goal_diff', 'hd_chance_diff']:
        for window in [10, 20]:
            h_col = f'h_{stat}_L{window}'
            a_col = f'a_{stat}_L{window}'
            if h_col in merged.columns and a_col in merged.columns:
                merged[f'{stat}_diff_L{window}'] = merged[h_col] - merged[a_col]
    
    return merged


def main():
    print("=" * 60)
    print("ADDING ADVANCED STATISTICS (FAST)")
    print("=" * 60)
    
    # Load and process
    df = load_data()
    df = calculate_team_rolling_stats(df)
    
    # Merge
    merged = merge_with_matchups(df)
    
    # Save
    output_path = DATA_DIR / 'processed' / 'matchups_with_advanced.csv'
    merged.to_csv(output_path, index=False)
    
    # Summary
    new_cols = [c for c in merged.columns if any(x in c for x in 
        ['shooting_pct', 'save_pct', 'faceoff', 'hitsFor', 'blockedShot', 
         'takeaway', 'giveaway', 'highDanger', 'hd_chance', 'penalty_diff', 'goal_diff_L'])]
    
    print(f"\nNew advanced stat features: {len(new_cols)}")
    print(f"Sample: {new_cols[:10]}")
    print(f"\nSaved to: {output_path}")
    print(f"Total features now: {len(merged.columns)}")
    
    return merged


if __name__ == '__main__':
    main()

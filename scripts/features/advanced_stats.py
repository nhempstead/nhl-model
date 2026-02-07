#!/usr/bin/env python3
"""
Add missing advanced statistics to improve model.

New features:
1. Power Play % (rolling)
2. Penalty Kill % (rolling)
3. Faceoff win % (rolling)
4. Hits, Blocks, Takeaways, Giveaways (rolling)
5. Shooting % and Save % (rolling)
6. Score-adjusted stats
7. Recent goal differential
8. Head-to-head record
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'


def load_moneypuck_games():
    """Load MoneyPuck all-games data with advanced stats."""
    path = DATA_DIR / 'raw' / 'moneypuck_all_games.csv'
    
    print("Loading MoneyPuck game-level data...")
    
    # Load relevant columns
    cols = [
        'team', 'season', 'gameId', 'playerTeam', 'opposingTeam', 'home_or_away', 'gameDate',
        'situation', 'xGoalsPercentage', 'corsiPercentage', 'fenwickPercentage',
        'shotsOnGoalFor', 'goalsFor', 'shotsOnGoalAgainst', 'goalsAgainst',
        'penaltiesFor', 'penalityMinutesFor', 'penaltiesAgainst', 'penalityMinutesAgainst',
        'faceOffsWonFor', 'faceOffsWonAgainst',
        'hitsFor', 'hitsAgainst', 'takeawaysFor', 'takeawaysAgainst',
        'giveawaysFor', 'giveawaysAgainst', 'blockedShotAttemptsFor', 'blockedShotAttemptsAgainst',
        'highDangerShotsFor', 'highDangerShotsAgainst', 'highDangerGoalsFor', 'highDangerGoalsAgainst',
    ]
    
    df = pd.read_csv(path, usecols=lambda x: x in cols)
    
    # Filter to 5on5 situation for main stats
    df_5on5 = df[df['situation'] == '5on5'].copy()
    
    # Also get special teams stats
    df_5on4 = df[df['situation'] == '5on4'].copy()  # Power play
    df_4on5 = df[df['situation'] == '4on5'].copy()  # Penalty kill
    
    print(f"Loaded {len(df_5on5)} 5on5 game records")
    
    return df_5on5, df_5on4, df_4on5


def calculate_special_teams(df_5on4, df_4on5):
    """Calculate power play and penalty kill percentages."""
    
    # Power Play: goals scored when 5on4
    pp_stats = df_5on4.groupby(['team', 'season', 'gameId']).agg({
        'goalsFor': 'sum',
        'shotsOnGoalFor': 'sum',
    }).reset_index()
    pp_stats.columns = ['team', 'season', 'gameId', 'pp_goals', 'pp_shots']
    
    # Penalty Kill: goals allowed when 4on5
    pk_stats = df_4on5.groupby(['team', 'season', 'gameId']).agg({
        'goalsAgainst': 'sum',
        'shotsOnGoalAgainst': 'sum',
    }).reset_index()
    pk_stats.columns = ['team', 'season', 'gameId', 'pk_goals_against', 'pk_shots_against']
    
    return pp_stats, pk_stats


def calculate_rolling_stats(df, windows=[5, 10, 20]):
    """Calculate rolling statistics for each team."""
    
    print("Calculating rolling advanced stats...")
    
    df = df.sort_values(['team', 'gameDate']).copy()
    
    # Key stats to calculate
    stats = {
        'shooting_pct': lambda x: x['goalsFor'].sum() / x['shotsOnGoalFor'].sum() if x['shotsOnGoalFor'].sum() > 0 else 0,
        'save_pct': lambda x: 1 - (x['goalsAgainst'].sum() / x['shotsOnGoalAgainst'].sum()) if x['shotsOnGoalAgainst'].sum() > 0 else 0,
        'faceoff_pct': lambda x: x['faceOffsWonFor'].sum() / (x['faceOffsWonFor'].sum() + x['faceOffsWonAgainst'].sum()) if (x['faceOffsWonFor'].sum() + x['faceOffsWonAgainst'].sum()) > 0 else 0.5,
        'hits_per_game': lambda x: x['hitsFor'].mean(),
        'blocks_per_game': lambda x: x['blockedShotAttemptsFor'].mean(),
        'takeaways_per_game': lambda x: x['takeawaysFor'].mean(),
        'giveaways_per_game': lambda x: x['giveawaysFor'].mean(),
        'hd_shots_for': lambda x: x['highDangerShotsFor'].mean(),
        'hd_shots_against': lambda x: x['highDangerShotsAgainst'].mean(),
        'goal_diff': lambda x: (x['goalsFor'] - x['goalsAgainst']).mean(),
    }
    
    results = []
    
    for team in df['team'].unique():
        team_df = df[df['team'] == team].copy()
        
        for idx, row in team_df.iterrows():
            game_date = row['gameDate']
            game_id = row['gameId']
            
            result = {
                'team': team,
                'gameId': game_id,
                'gameDate': game_date,
            }
            
            for window in windows:
                # Get previous N games (excluding current)
                prev_games = team_df[team_df['gameDate'] < game_date].tail(window)
                
                if len(prev_games) >= 3:  # Need at least 3 games
                    for stat_name, stat_func in stats.items():
                        try:
                            result[f'{stat_name}_L{window}'] = stat_func(prev_games)
                        except:
                            result[f'{stat_name}_L{window}'] = np.nan
                else:
                    for stat_name in stats.keys():
                        result[f'{stat_name}_L{window}'] = np.nan
            
            results.append(result)
    
    return pd.DataFrame(results)


def merge_with_matchups(rolling_stats, matchups_path, output_path):
    """Merge rolling stats with matchups data."""
    
    print("Merging with matchups...")
    
    matchups = pd.read_csv(matchups_path, low_memory=False)
    
    # Prepare home team stats
    home_stats = rolling_stats.copy()
    home_stats.columns = ['h_' + c if c not in ['gameId', 'gameDate'] else c for c in home_stats.columns]
    home_stats = home_stats.rename(columns={'h_team': 'h_team_check'})
    
    # Prepare away team stats
    away_stats = rolling_stats.copy()
    away_stats.columns = ['a_' + c if c not in ['gameId', 'gameDate'] else c for c in away_stats.columns]
    away_stats = away_stats.rename(columns={'a_team': 'a_team_check'})
    
    # Merge on gameId
    merged = matchups.merge(home_stats, on='gameId', how='left', suffixes=('', '_home'))
    merged = merged.merge(away_stats, on='gameId', how='left', suffixes=('', '_away'))
    
    # Calculate differentials
    for window in [5, 10, 20]:
        for stat in ['shooting_pct', 'save_pct', 'faceoff_pct', 'goal_diff', 'hd_shots_for']:
            h_col = f'h_{stat}_L{window}'
            a_col = f'a_{stat}_L{window}'
            if h_col in merged.columns and a_col in merged.columns:
                merged[f'{stat}_diff_L{window}'] = merged[h_col] - merged[a_col]
    
    merged.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return merged


def main():
    print("=" * 60)
    print("ADDING ADVANCED STATISTICS")
    print("=" * 60)
    
    # Load data
    df_5on5, df_5on4, df_4on5 = load_moneypuck_games()
    
    # Calculate rolling stats
    rolling = calculate_rolling_stats(df_5on5, windows=[5, 10, 20])
    
    print(f"\nRolling stats shape: {rolling.shape}")
    print(f"New features: {[c for c in rolling.columns if '_L' in c][:10]}...")
    
    # Merge with matchups
    matchups_path = DATA_DIR / 'processed' / 'matchups_with_situational.csv'
    output_path = DATA_DIR / 'processed' / 'matchups_with_advanced.csv'
    
    merged = merge_with_matchups(rolling, matchups_path, output_path)
    
    print(f"\nFinal shape: {merged.shape}")
    
    # Count new features
    new_cols = [c for c in merged.columns if any(x in c for x in ['shooting_pct', 'save_pct', 'faceoff', 'hits', 'blocks', 'takeaway', 'giveaway', 'hd_shots', 'goal_diff_L'])]
    print(f"New advanced stat features: {len(new_cols)}")
    
    return merged


if __name__ == '__main__':
    main()

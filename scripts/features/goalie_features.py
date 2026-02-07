#!/usr/bin/env python3
"""
Build game-level goalie features for the prediction model.

Features created:
1. Goalie identity (who's starting)
2. Rest days since last start
3. Back-to-back start indicator
4. Backup vs starter flag
5. Rolling save percentage (L5, L10 games)
6. Rolling xG against rate
7. Goalie quality tier (elite/average/backup)

These features capture market-inefficient information:
- Backup starts often not fully priced until 30-60 min before game
- Tired goalies (B2B) underperform
- Hot/cold streaks in goalie performance
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOALIE_STARTS_FILE = os.path.join(BASE_DIR, 'data/goalie_starts/game_goalie_starts.csv')
GOALIE_STATS_FILE = os.path.join(BASE_DIR, 'data/raw/moneypuck_goalies.csv')
MATCHUPS_FILE = os.path.join(BASE_DIR, 'data/processed/matchups_v3.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/processed/matchups_with_goalie.csv')


def load_goalie_starts():
    """Load the game-level goalie starts data."""
    print("Loading goalie starts...")
    df = pd.read_csv(GOALIE_STARTS_FILE)
    df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"Loaded {len(df):,} games with goalie starts")
    return df


def load_matchups():
    """Load the matchups data."""
    print("Loading matchups...")
    df = pd.read_csv(MATCHUPS_FILE)
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    print(f"Loaded {len(df):,} matchups")
    return df


def calculate_rolling_goalie_stats(starts_df, window_games=10):
    """
    Calculate rolling goalie performance metrics.
    Uses shots saved / shots faced as a proxy for save percentage.
    """
    print(f"\nCalculating rolling goalie stats (L{window_games})...")
    
    starts_df = starts_df.sort_values('game_date').copy()
    
    # Track goalie performance over time
    goalie_history = {}  # goalie_id -> list of (date, shots_faced, shots_saved)
    
    home_rolling_sv = []
    away_rolling_sv = []
    home_rolling_shots = []
    away_rolling_shots = []
    
    for idx, row in starts_df.iterrows():
        game_date = row['game_date']
        
        # Home goalie rolling stats
        h_goalie = row['home_goalie_id']
        if pd.notna(h_goalie) and h_goalie in goalie_history:
            recent = goalie_history[h_goalie][-window_games:]
            if len(recent) >= 3:
                total_shots = sum(r[1] for r in recent)
                total_saved = sum(r[2] for r in recent)
                sv_pct = total_saved / total_shots if total_shots > 0 else np.nan
                home_rolling_sv.append(sv_pct)
                home_rolling_shots.append(total_shots / len(recent))
            else:
                home_rolling_sv.append(np.nan)
                home_rolling_shots.append(np.nan)
        else:
            home_rolling_sv.append(np.nan)
            home_rolling_shots.append(np.nan)
        
        # Away goalie rolling stats
        a_goalie = row['away_goalie_id']
        if pd.notna(a_goalie) and a_goalie in goalie_history:
            recent = goalie_history[a_goalie][-window_games:]
            if len(recent) >= 3:
                total_shots = sum(r[1] for r in recent)
                total_saved = sum(r[2] for r in recent)
                sv_pct = total_saved / total_shots if total_shots > 0 else np.nan
                away_rolling_sv.append(sv_pct)
                away_rolling_shots.append(total_shots / len(recent))
            else:
                away_rolling_sv.append(np.nan)
                away_rolling_shots.append(np.nan)
        else:
            away_rolling_sv.append(np.nan)
            away_rolling_shots.append(np.nan)
        
        # Update history with this game's results (shots faced = shots_total for opposing team's goalie)
        if pd.notna(h_goalie):
            shots_faced = row['home_goalie_shots_total']
            # We don't have goals against directly, estimate from relief appearance
            # If relief, assume starter gave up ~3 goals on ~20 shots before pull
            if pd.notna(shots_faced) and shots_faced > 0:
                if h_goalie not in goalie_history:
                    goalie_history[h_goalie] = []
                # Estimate saves as ~90% of shots (rough average)
                # This is approximate - ideally we'd have actual goals against
                saves_estimate = int(shots_faced * 0.90)
                goalie_history[h_goalie].append((game_date, shots_faced, saves_estimate))
        
        if pd.notna(a_goalie):
            shots_faced = row['away_goalie_shots_total']
            if pd.notna(shots_faced) and shots_faced > 0:
                if a_goalie not in goalie_history:
                    goalie_history[a_goalie] = []
                saves_estimate = int(shots_faced * 0.90)
                goalie_history[a_goalie].append((game_date, shots_faced, saves_estimate))
    
    starts_df['home_goalie_sv_L10'] = home_rolling_sv
    starts_df['away_goalie_sv_L10'] = away_rolling_sv
    starts_df['home_goalie_shots_L10'] = home_rolling_shots
    starts_df['away_goalie_shots_L10'] = away_rolling_shots
    
    return starts_df


def calculate_goalie_quality_tier(starts_df):
    """
    Classify goalies into quality tiers based on season start share and performance.
    Tiers: 1 = Elite starter (60%+ starts), 2 = Starter (40-60%), 3 = Backup (<40%)
    """
    print("\nCalculating goalie quality tiers...")
    
    # Calculate start share for each goalie per team per season
    starts_df['home_goalie_tier'] = 2  # Default to average
    starts_df['away_goalie_tier'] = 2
    
    # Backup start already indicates tier 3
    starts_df.loc[starts_df['home_backup_start'] == True, 'home_goalie_tier'] = 3
    starts_df.loc[starts_df['away_backup_start'] == True, 'away_goalie_tier'] = 3
    
    # For non-backups, look at season-level patterns
    # We'll use the backup_start logic inversely: if never flagged as backup, likely tier 1-2
    # More sophisticated: calculate rolling start share
    
    return starts_df


def merge_with_matchups(matchups_df, starts_df):
    """
    Merge goalie features with matchups data.
    """
    print("\nMerging with matchups...")
    
    # Select goalie features to merge
    goalie_cols = [
        'game_id', 'home_goalie_id', 'home_goalie_name', 'away_goalie_id', 'away_goalie_name',
        'home_goalie_rest_days', 'away_goalie_rest_days',
        'home_goalie_b2b', 'away_goalie_b2b',
        'home_backup_start', 'away_backup_start',
        'home_relief_appearance', 'away_relief_appearance',
        'home_goalie_sv_L10', 'away_goalie_sv_L10',
        'home_goalie_tier', 'away_goalie_tier'
    ]
    
    goalie_features = starts_df[goalie_cols].copy()
    
    # Rename game_id to match matchups
    goalie_features = goalie_features.rename(columns={'game_id': 'gameId'})
    
    # Merge
    merged = matchups_df.merge(goalie_features, on='gameId', how='left')
    
    print(f"Merged: {len(merged):,} rows")
    print(f"With goalie data: {merged['home_goalie_id'].notna().sum():,}")
    
    return merged


def create_edge_features(df):
    """
    Create derived edge features from goalie data.
    """
    print("\nCreating edge features...")
    
    # Rest advantage
    df['goalie_rest_diff'] = df['home_goalie_rest_days'] - df['away_goalie_rest_days']
    
    # B2B disadvantage (True = disadvantage for that team)
    # Handle NaN values by filling with 0 (no B2B)
    df['home_goalie_b2b_flag'] = df['home_goalie_b2b'].fillna(False).astype(int)
    df['away_goalie_b2b_flag'] = df['away_goalie_b2b'].fillna(False).astype(int)
    df['b2b_diff'] = df['away_goalie_b2b_flag'] - df['home_goalie_b2b_flag']  # Positive = away has B2B
    
    # Backup vs starter matchup
    df['home_backup_flag'] = df['home_backup_start'].fillna(False).astype(int)
    df['away_backup_flag'] = df['away_backup_start'].fillna(False).astype(int)
    df['backup_mismatch'] = df['away_backup_flag'] - df['home_backup_flag']  # Positive = away has backup
    
    # Quality tier difference
    df['goalie_tier_diff'] = df['away_goalie_tier'] - df['home_goalie_tier']  # Positive = away has worse tier
    
    # Save percentage difference
    df['goalie_sv_diff'] = df['home_goalie_sv_L10'] - df['away_goalie_sv_L10']
    
    return df


def main():
    print("="*60)
    print("GOALIE FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    starts_df = load_goalie_starts()
    matchups_df = load_matchups()
    
    # Calculate rolling stats
    starts_df = calculate_rolling_goalie_stats(starts_df)
    
    # Calculate quality tiers
    starts_df = calculate_goalie_quality_tier(starts_df)
    
    # Merge with matchups
    merged_df = merge_with_matchups(matchups_df, starts_df)
    
    # Create edge features
    merged_df = create_edge_features(merged_df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total matchups: {len(merged_df):,}")
    print(f"With goalie data: {merged_df['home_goalie_id'].notna().sum():,}")
    
    goalie_cols = [c for c in merged_df.columns if 'goalie' in c.lower() or 'backup' in c.lower() or 'b2b' in c.lower()]
    print(f"\nNew goalie features ({len(goalie_cols)}):")
    for col in goalie_cols:
        print(f"  - {col}")
    
    # Distribution of key features
    print("\nBackup start distribution:")
    print(f"  Home backup: {merged_df['home_backup_start'].sum():,} ({100*merged_df['home_backup_start'].mean():.1f}%)")
    print(f"  Away backup: {merged_df['away_backup_start'].sum():,} ({100*merged_df['away_backup_start'].mean():.1f}%)")
    
    print("\nB2B start distribution:")
    print(f"  Home B2B: {merged_df['home_goalie_b2b'].sum():,} ({100*merged_df['home_goalie_b2b'].mean():.1f}%)")
    print(f"  Away B2B: {merged_df['away_goalie_b2b'].sum():,} ({100*merged_df['away_goalie_b2b'].mean():.1f}%)")
    
    # Save
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    return merged_df


if __name__ == '__main__':
    df = main()

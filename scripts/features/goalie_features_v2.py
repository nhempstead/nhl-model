#!/usr/bin/env python3
"""
Goalie Features V2 - Build historical goalie quality metrics for model training.

Key features:
1. GSAx (Goals Saved Above Expected) - rolling window
2. Save percentage - rolling window  
3. Quality starts percentage
4. Games played (workload/fatigue)
5. Career baseline (regression target)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def load_goalie_data():
    """Load MoneyPuck goalie data."""
    path = DATA_DIR / 'raw' / 'moneypuck_goalies.csv'
    if not path.exists():
        print(f"❌ Goalie data not found: {path}")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} goalie records")
    return df


def calculate_gsax(row):
    """Calculate Goals Saved Above Expected."""
    xgoals = row.get('xGoals', 0) or 0
    goals = row.get('goals', 0) or 0
    return xgoals - goals


def calculate_sv_pct(row):
    """Calculate save percentage."""
    shots = row.get('ongoal', 0) or 0
    goals = row.get('goals', 0) or 0
    if shots > 0:
        return 1 - (goals / shots)
    return np.nan


def build_goalie_ratings(df):
    """
    Build goalie quality ratings.
    
    Rating scale (0-100):
    - 0-30: Replacement level / AHL
    - 30-50: Below average NHL
    - 50-70: Average starter
    - 70-85: Good starter
    - 85-100: Elite
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to 'all' situations and reasonable sample
    goalies = df[
        (df['situation'] == 'all') & 
        (df['games_played'] >= 5)
    ].copy()
    
    if goalies.empty:
        goalies = df[df['games_played'] >= 5].copy()
    
    # Calculate metrics
    goalies['gsax'] = goalies.apply(calculate_gsax, axis=1)
    goalies['sv_pct'] = goalies.apply(calculate_sv_pct, axis=1)
    goalies['gsax_per_game'] = goalies['gsax'] / goalies['games_played'].clip(lower=1)
    
    # Rating formula
    def calc_rating(row):
        sv_pct = row['sv_pct']
        gsax_pg = row['gsax_per_game']
        games = row['games_played']
        
        if pd.isna(sv_pct):
            return 50  # Default
        
        # SV% component: baseline .910, each .001 = 1 point
        sv_rating = 50 + (sv_pct - 0.910) * 1000
        
        # GSAx component: each 0.1 GSAx/game = 5 points
        gsax_rating = gsax_pg * 50
        
        # Weight by sample size
        if games < 10:
            weight = 0.3
        elif games < 20:
            weight = 0.5
        else:
            weight = 0.7
        
        rating = sv_rating * (1 - weight * 0.3) + gsax_rating * weight * 0.3 + 50 * (1 - weight)
        
        return max(10, min(95, rating))
    
    goalies['rating'] = goalies.apply(calc_rating, axis=1)
    
    # Aggregate to team level (primary starter)
    team_goalies = goalies.sort_values('games_played', ascending=False).groupby('team').first().reset_index()
    
    return goalies, team_goalies


def build_historical_goalie_features(matchups_df, goalie_df):
    """
    Add goalie features to historical matchups.
    
    For each game, we need:
    - Home goalie rating (pre-game)
    - Away goalie rating (pre-game)
    - Goalie differential
    - Backup indicator (if available)
    """
    # This requires historical goalie starts data
    # For now, use season-level goalie ratings as proxy
    
    print("⚠️  Historical goalie starts not available - using season-level ratings")
    
    # Group goalie stats by season
    if 'season' not in goalie_df.columns:
        print("No season column in goalie data - skipping")
        return matchups_df
    
    # Build season-level team goalie ratings
    season_ratings = {}
    for season in goalie_df['season'].unique():
        season_data = goalie_df[goalie_df['season'] == season]
        _, team_ratings = build_goalie_ratings(season_data)
        for _, row in team_ratings.iterrows():
            key = (season, row['team'])
            season_ratings[key] = row['rating']
    
    # Add to matchups
    def get_goalie_rating(row, team_col):
        team = row[team_col]
        season = row['season']
        return season_ratings.get((season, team), 50)
    
    matchups_df['h_goalie_rating'] = matchups_df.apply(lambda r: get_goalie_rating(r, 'h_team'), axis=1)
    matchups_df['a_goalie_rating'] = matchups_df.apply(lambda r: get_goalie_rating(r, 'a_team'), axis=1)
    matchups_df['goalie_rating_diff'] = matchups_df['h_goalie_rating'] - matchups_df['a_goalie_rating']
    
    return matchups_df


def get_current_goalie_ratings():
    """Get current season goalie ratings for prediction."""
    path = DATA_DIR / 'raw' / 'moneypuck_goalies_current.csv'
    if not path.exists():
        print(f"❌ Current goalie data not found: {path}")
        return {}
    
    df = pd.read_csv(path)
    goalies, team_goalies = build_goalie_ratings(df)
    
    ratings = {}
    for _, row in team_goalies.iterrows():
        ratings[row['team']] = {
            'goalie': row.get('name', 'Unknown'),
            'rating': round(row['rating'], 1),
            'sv_pct': round(row['sv_pct'], 3) if pd.notna(row['sv_pct']) else None,
            'gsax': round(row['gsax'], 2),
            'games': int(row['games_played']),
        }
    
    return ratings


def main():
    print("=" * 60)
    print("GOALIE FEATURES V2")
    print("=" * 60)
    
    # Current ratings
    print("\n📊 Current Season Goalie Ratings:")
    ratings = get_current_goalie_ratings()
    
    # Sort by rating
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1]['rating'], reverse=True)
    
    print(f"\n{'Team':<5} {'Goalie':<20} {'Rating':>7} {'SV%':>7} {'GSAx':>7} {'GP':>4}")
    print("-" * 55)
    for team, data in sorted_ratings[:15]:
        sv = f"{data['sv_pct']:.3f}" if data['sv_pct'] else "N/A"
        print(f"{team:<5} {data['goalie']:<20} {data['rating']:>7.1f} {sv:>7} {data['gsax']:>7.2f} {data['games']:>4}")
    
    print("\n... (showing top 15)")
    
    # Save ratings
    output_path = DATA_DIR / 'ratings' / 'goalie_ratings.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'generated': pd.Timestamp.now().isoformat(),
            'ratings': ratings
        }, f, indent=2)
    
    print(f"\n✅ Saved to {output_path}")
    
    return ratings


if __name__ == '__main__':
    main()

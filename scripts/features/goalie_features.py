#!/usr/bin/env python3
"""
Goaltender features for NHL win probability model.

Key insight from MoneyPuck: Goaltending accounts for 29% of game outcome.
This module calculates GSAx (Goals Saved Above Expected) and other goalie metrics.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')


def load_goalie_data():
    """Load MoneyPuck goalie data"""
    path = os.path.join(DATA_DIR, 'raw/moneypuck_goalies.csv')
    df = pd.read_csv(path)
    
    # Filter to 5on5 situation (most predictive)
    df = df[df['situation'] == '5on5'].copy()
    
    print(f"Loaded {len(df)} goalie seasons")
    return df


def calculate_gsax(df):
    """
    Calculate Goals Saved Above Expected (GSAx)
    
    GSAx = xGoals - Goals Allowed
    Positive = saved more than expected (good)
    Negative = allowed more than expected (bad)
    
    Also calculate per-60 rate for comparison across different ice times.
    """
    
    # GSAx = expected goals - actual goals
    df['gsax'] = df['xGoals'] - df['goals']
    
    # Per 60 minutes rate
    df['icetime_hours'] = df['icetime'] / 3600
    df['gsax_per60'] = (df['gsax'] / df['icetime_hours']) * 60
    
    # Save percentage (traditional metric)
    df['shots_faced'] = df['ongoal']
    df['saves'] = df['shots_faced'] - df['goals']
    df['save_pct'] = df['saves'] / df['shots_faced'].replace(0, 1)
    
    # Expected save percentage
    df['xsave_pct'] = (df['xOnGoal'] - df['xGoals']) / df['xOnGoal'].replace(0, 1)
    
    # Save pct above expected
    df['save_pct_above_expected'] = df['save_pct'] - df['xsave_pct']
    
    return df


def get_goalie_ratings(min_games=10):
    """
    Get current goalie ratings with regression to mean.
    
    Small sample sizes are regressed toward league average.
    """
    
    df = load_goalie_data()
    df = calculate_gsax(df)
    
    # Group by goalie and season
    goalie_seasons = df.groupby(['playerId', 'name', 'team', 'season']).agg({
        'games_played': 'sum',
        'icetime': 'sum',
        'xGoals': 'sum',
        'goals': 'sum',
        'ongoal': 'sum',
        'gsax': 'sum',
    }).reset_index()
    
    # Recalculate rates
    goalie_seasons['gsax_per60'] = (goalie_seasons['gsax'] / (goalie_seasons['icetime'] / 3600)) * 60
    goalie_seasons['save_pct'] = (goalie_seasons['ongoal'] - goalie_seasons['goals']) / goalie_seasons['ongoal'].replace(0, 1)
    
    # Filter to meaningful sample
    goalie_seasons = goalie_seasons[goalie_seasons['games_played'] >= min_games]
    
    # Regression to mean
    # More games = trust individual rate more
    # Fewer games = regress to league average
    league_avg_gsax_per60 = goalie_seasons['gsax_per60'].mean()
    
    # Regression factor: games / (games + regression_constant)
    regression_constant = 25  # ~25 games to trust 50% of observed rate
    goalie_seasons['regression_factor'] = goalie_seasons['games_played'] / (goalie_seasons['games_played'] + regression_constant)
    
    goalie_seasons['gsax_per60_regressed'] = (
        goalie_seasons['regression_factor'] * goalie_seasons['gsax_per60'] +
        (1 - goalie_seasons['regression_factor']) * league_avg_gsax_per60
    )
    
    print(f"Calculated ratings for {len(goalie_seasons)} goalie-seasons")
    
    return goalie_seasons


def get_current_goalie_ratings(season=2025):
    """Get most recent ratings for each goalie"""
    
    ratings = get_goalie_ratings()
    
    # Get most recent season for each goalie
    current = ratings[ratings['season'] == season].copy()
    
    # For goalies without current season data, use previous season
    if len(current) == 0:
        current = ratings[ratings['season'] == season - 1].copy()
    
    # Sort by GSAx
    current = current.sort_values('gsax_per60_regressed', ascending=False)
    
    return current


def get_team_goalie_strength(team, season=2025):
    """
    Get combined goaltending strength for a team.
    
    Weights by games played (starter vs backup).
    """
    
    ratings = get_goalie_ratings()
    
    # Get team's goalies for current season
    team_goalies = ratings[(ratings['team'] == team) & (ratings['season'] == season)]
    
    if len(team_goalies) == 0:
        # Try previous season
        team_goalies = ratings[(ratings['team'] == team) & (ratings['season'] == season - 1)]
    
    if len(team_goalies) == 0:
        return {'gsax_per60': 0, 'starter': 'Unknown', 'starter_gsax': 0}
    
    # Weight by games played
    total_games = team_goalies['games_played'].sum()
    weighted_gsax = (team_goalies['gsax_per60_regressed'] * team_goalies['games_played']).sum() / total_games
    
    # Identify starter (most games)
    starter = team_goalies.loc[team_goalies['games_played'].idxmax()]
    
    return {
        'gsax_per60': weighted_gsax,
        'starter': starter['name'],
        'starter_gsax': starter['gsax_per60_regressed'],
        'starter_games': starter['games_played'],
    }


def calculate_goalie_matchup_adjustment(home_team, away_team, season=2025):
    """
    Calculate win probability adjustment based on goaltending.
    
    Returns adjustment to add to base win probability.
    Scale: ±5% for elite vs bad goalie matchup.
    """
    
    home_goalie = get_team_goalie_strength(home_team, season)
    away_goalie = get_team_goalie_strength(away_team, season)
    
    # GSAx differential
    gsax_diff = home_goalie['gsax_per60'] - away_goalie['gsax_per60']
    
    # Convert to win probability adjustment
    # Rough scale: 1 GSAx/60 difference ≈ 2-3% win probability
    adjustment = gsax_diff * 0.025
    
    # Cap at ±5%
    adjustment = max(min(adjustment, 0.05), -0.05)
    
    return {
        'adjustment': adjustment,
        'home_goalie': home_goalie,
        'away_goalie': away_goalie,
        'gsax_diff': gsax_diff,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("GOALIE RATINGS")
    print("=" * 60)
    
    ratings = get_current_goalie_ratings()
    
    print("\nTop 10 Goalies (by regressed GSAx/60):")
    print(ratings[['name', 'team', 'games_played', 'gsax_per60', 'gsax_per60_regressed', 'save_pct']].head(10).to_string())
    
    print("\n\nBottom 10 Goalies:")
    print(ratings[['name', 'team', 'games_played', 'gsax_per60', 'gsax_per60_regressed', 'save_pct']].tail(10).to_string())
    
    # Example matchup
    print("\n" + "=" * 60)
    print("EXAMPLE MATCHUP: DAL vs STL")
    print("=" * 60)
    matchup = calculate_goalie_matchup_adjustment('DAL', 'STL')
    print(f"DAL goalie: {matchup['home_goalie']['starter']} (GSAx/60: {matchup['home_goalie']['gsax_per60']:.2f})")
    print(f"STL goalie: {matchup['away_goalie']['starter']} (GSAx/60: {matchup['away_goalie']['gsax_per60']:.2f})")
    print(f"GSAx differential: {matchup['gsax_diff']:.2f}")
    print(f"Win prob adjustment for DAL: {matchup['adjustment']:+.1%}")

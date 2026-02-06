#!/usr/bin/env python3
"""
Build benchmark dataset with historical odds for model calibration.

Creates a clean dataset where we can compare model predictions to market odds.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'


def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def prob_to_american(prob):
    """Convert probability to American odds."""
    if pd.isna(prob) or prob <= 0 or prob >= 1:
        return np.nan
    if prob >= 0.5:
        return -round(prob / (1 - prob) * 100)
    return round((1 - prob) / prob * 100)


def estimate_underdog_odds(fav_ml, vig=1.04):
    """Estimate underdog odds from favorite ML assuming ~4% vig."""
    fav_prob = american_to_prob(fav_ml)
    if pd.isna(fav_prob):
        return np.nan
    dog_prob = vig - fav_prob
    return prob_to_american(dog_prob)


def build_benchmark():
    """Build benchmark dataset from historical odds."""
    
    # Load data
    path = DATA_DIR / 'historical_odds' / 'nhl_data_extensive.csv'
    df = pd.read_csv(path)
    
    # Filter to games with real moneylines (not placeholder -105)
    df = df[df['favorite_moneyline'] != -105].copy()
    print(f"Games with real odds: {len(df)//2}")
    
    # Process into game-level data
    home = df[df['is_home'] == True][['game_id', 'date', 'team_name', 'won', 'favorite_moneyline', 'over_under']].copy()
    away = df[df['is_home'] == False][['game_id', 'team_name', 'won']].copy()
    
    home.columns = ['game_id', 'date', 'home_team', 'home_won', 'fav_ml', 'total']
    away.columns = ['game_id', 'away_team', 'away_won']
    
    games = home.merge(away, on='game_id')
    
    # Determine which team is the favorite
    # Heuristic: if fav_ml is very negative (heavy favorite), likely the home team
    # But we can't be 100% sure from this data
    # Let's assume: when fav_ml < -130, home is likely favorite
    #               when fav_ml > -130, could go either way
    
    # Better approach: use the spread if available
    # In NHL, favorite is usually at -1.5 spread
    # Since both teams show -1.5 (data issue), use other signals
    
    # For now, create two scenarios:
    # 1. Assume home is always the favorite (use fav_ml for home)
    # 2. Calculate expected calibration
    
    games['fav_prob'] = games['fav_ml'].apply(american_to_prob)
    games['dog_ml_est'] = games['fav_ml'].apply(estimate_underdog_odds)
    games['dog_prob'] = games['dog_ml_est'].apply(american_to_prob)
    
    # Calculate no-vig probabilities
    games['total_implied'] = games['fav_prob'] + games['dog_prob']
    games['fav_prob_novig'] = games['fav_prob'] / games['total_implied']
    games['dog_prob_novig'] = games['dog_prob'] / games['total_implied']
    
    # Save
    output_path = DATA_DIR / 'benchmark' / 'games_with_odds.csv'
    output_path.parent.mkdir(exist_ok=True)
    games.to_csv(output_path, index=False)
    
    print(f"\nSaved {len(games)} games to {output_path}")
    print(f"Date range: {games['date'].min()} to {games['date'].max()}")
    
    # Calculate benchmark calibration
    print("\n" + "="*60)
    print("BENCHMARK: Market Calibration")
    print("="*60)
    
    # Bin by favorite probability
    games['prob_bin'] = pd.cut(games['fav_prob_novig'], 
                                bins=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0])
    
    # Note: we're comparing fav win rate to fav prob, not home win rate
    # This requires knowing which team is the favorite
    # For now, approximate by looking at negative moneylines
    games['likely_home_fav'] = games['fav_ml'] < -130  # Heavy favorites likely home
    
    # When home is likely favorite, home_won = fav_won
    # When away is likely favorite, away_won = fav_won
    games['fav_won'] = games.apply(
        lambda r: r['home_won'] if r['fav_ml'] < 0 else (1 - r['home_won']),
        axis=1
    )
    
    calibration = games.groupby('prob_bin').agg({
        'fav_won': ['mean', 'count'],
        'fav_prob_novig': 'mean'
    })
    calibration.columns = ['actual', 'count', 'implied']
    calibration['diff'] = calibration['actual'] - calibration['implied']
    
    print("\nCalibration (implied vs actual):")
    print(calibration.round(3))
    
    # Brier score
    games['brier'] = (games['fav_won'] - games['fav_prob_novig']) ** 2
    brier = games['brier'].mean()
    print(f"\nMarket Brier Score: {brier:.4f}")
    
    return games


if __name__ == '__main__':
    games = build_benchmark()

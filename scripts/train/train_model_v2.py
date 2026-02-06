#!/usr/bin/env python3
"""
Train NHL Model V2 - With goalie and schedule features.

Target: Beat market benchmark (Brier 0.2248, Log Loss 0.6367)

Key improvements over V1:
1. Goalie features (GSAx-based ratings)
2. Schedule features (B2B, rest, travel)
3. Calibration against market closing odds
4. Walk-forward validation on recent seasons
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'trained'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Market benchmark to beat
MARKET_BRIER = 0.2248
MARKET_LOGLOSS = 0.6367


def load_data():
    """Load processed matchup data with all features."""
    # Try full-featured first
    full_path = DATA_DIR / 'processed' / 'matchups_with_all_features.csv'
    schedule_path = DATA_DIR / 'processed' / 'matchups_with_schedule.csv'
    
    if full_path.exists():
        df = pd.read_csv(full_path)
        print(f"Loaded full-featured data: {len(df)} games")
    elif schedule_path.exists():
        df = pd.read_csv(schedule_path)
        print(f"Loaded schedule-enhanced data: {len(df)} games")
    else:
        df = pd.read_csv(DATA_DIR / 'processed' / 'matchups_featured.csv')
        print(f"Loaded base data: {len(df)} games")
    
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    return df


def load_market_odds():
    """Load historical market odds for calibration benchmark."""
    odds_path = DATA_DIR / 'historical_odds_api' / 'nhl_odds_all.csv'
    if not odds_path.exists():
        print("Warning: No historical odds found")
        return None
    
    odds = pd.read_csv(odds_path)
    
    def american_to_prob(o):
        if pd.isna(o) or o == 0: return 0.5
        return 100/(o+100) if o > 0 else abs(o)/(abs(o)+100)
    
    odds['home_prob'] = odds['home_odds'].apply(american_to_prob)
    odds['away_prob'] = odds['away_odds'].apply(american_to_prob)
    odds['market_home_prob'] = odds['home_prob'] / (odds['home_prob'] + odds['away_prob'])
    odds['date'] = pd.to_datetime(odds['commence_time']).dt.date
    
    # Team name mapping
    name_to_abbrev = {
        'Anaheim Ducks': 'ANA', 'Arizona Coyotes': 'ARI', 'Boston Bruins': 'BOS',
        'Buffalo Sabres': 'BUF', 'Calgary Flames': 'CGY', 'Carolina Hurricanes': 'CAR',
        'Chicago Blackhawks': 'CHI', 'Colorado Avalanche': 'COL', 'Columbus Blue Jackets': 'CBJ',
        'Dallas Stars': 'DAL', 'Detroit Red Wings': 'DET', 'Edmonton Oilers': 'EDM',
        'Florida Panthers': 'FLA', 'Los Angeles Kings': 'LAK', 'Minnesota Wild': 'MIN',
        'Montréal Canadiens': 'MTL', 'Montreal Canadiens': 'MTL', 'Nashville Predators': 'NSH',
        'New Jersey Devils': 'NJD', 'New York Islanders': 'NYI', 'New York Rangers': 'NYR',
        'Ottawa Senators': 'OTT', 'Philadelphia Flyers': 'PHI', 'Pittsburgh Penguins': 'PIT',
        'San Jose Sharks': 'SJS', 'Seattle Kraken': 'SEA', 'St. Louis Blues': 'STL',
        'Tampa Bay Lightning': 'TBL', 'Toronto Maple Leafs': 'TOR', 'Vancouver Canucks': 'VAN',
        'Vegas Golden Knights': 'VGK', 'Washington Capitals': 'WSH', 'Winnipeg Jets': 'WPG',
        'Utah Hockey Club': 'UTA',
    }
    odds['h_team'] = odds['home_team'].map(name_to_abbrev)
    odds['a_team'] = odds['away_team'].map(name_to_abbrev)
    
    return odds[['date', 'h_team', 'a_team', 'market_home_prob']]


def get_feature_cols(df):
    """Identify feature columns including new schedule features."""
    exclude = ['gameId', 'gameDate', 'season', 'h_team', 'a_team', 'home_win',
               'h_totalGoalsFor', 'h_totalGoalsAgainst', 'a_totalGoalsFor', 'a_totalGoalsAgainst',
               'date', 'market_home_prob']  # Exclude market prob (that's what we're trying to beat)
    
    return [c for c in df.columns if c not in exclude and not c.startswith('Unnamed')]


def train_xgboost_v2(X_train, y_train, X_val, y_val):
    """XGBoost with tuned parameters for probability estimation."""
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=20,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'train_brier': brier_score_loss(y_train, train_pred),
        'val_brier': brier_score_loss(y_val, val_pred),
        'val_accuracy': accuracy_score(y_val, (val_pred > 0.5).astype(int)),
        'val_logloss': log_loss(y_val, val_pred),
    }
    
    return model, metrics, val_pred


def calibrate_probabilities(model, X_val, y_val, val_probs):
    """Isotonic regression calibration for better probability estimates."""
    from sklearn.isotonic import IsotonicRegression
    
    # Fit isotonic regression on validation
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(val_probs, y_val)
    
    cal_probs = iso.predict(val_probs)
    
    cal_metrics = {
        'cal_brier': brier_score_loss(y_val, cal_probs),
        'cal_logloss': log_loss(y_val, cal_probs),
    }
    
    return iso, cal_metrics


def evaluate_vs_market(df_test, y_test, model_probs, market_odds):
    """Compare model predictions to market closing odds."""
    if market_odds is None:
        return None
    
    # Merge test data with market odds
    df_test = df_test.copy()
    df_test['date'] = pd.to_datetime(df_test['gameDate']).dt.date
    df_test['model_prob'] = model_probs
    df_test['actual'] = y_test.values
    
    merged = df_test.merge(market_odds, on=['date', 'h_team', 'a_team'], how='inner')
    
    if len(merged) == 0:
        print("Warning: No games matched with market odds")
        return None
    
    # Calculate metrics for both
    model_brier = brier_score_loss(merged['actual'], merged['model_prob'])
    market_brier = brier_score_loss(merged['actual'], merged['market_home_prob'])
    
    model_ll = log_loss(merged['actual'], merged['model_prob'].clip(0.01, 0.99))
    market_ll = log_loss(merged['actual'], merged['market_home_prob'].clip(0.01, 0.99))
    
    return {
        'n_games': len(merged),
        'model_brier': model_brier,
        'market_brier': market_brier,
        'brier_diff': model_brier - market_brier,
        'model_logloss': model_ll,
        'market_logloss': market_ll,
        'logloss_diff': model_ll - market_ll,
    }


def main():
    print("=" * 70)
    print("NHL MODEL V2 TRAINING")
    print(f"Target: Beat market Brier {MARKET_BRIER:.4f}, LogLoss {MARKET_LOGLOSS:.4f}")
    print("=" * 70)
    
    # Load data
    df = load_data()
    market_odds = load_market_odds()
    
    feature_cols = get_feature_cols(df)
    print(f"\nFeatures: {len(feature_cols)}")
    
    # Check for new features
    schedule_features = [c for c in feature_cols if any(x in c for x in ['b2b', 'rest', 'travel', 'tz_change', 'fatigue'])]
    goalie_features = [c for c in feature_cols if 'goalie' in c.lower()]
    print(f"Schedule features: {schedule_features}")
    print(f"Goalie features: {goalie_features}")
    
    # Walk-forward split - train on older, test on recent
    # Train: 2008-2022, Val: 2023, Test: 2024-2025
    train_seasons = list(range(2008, 2023))
    val_seasons = [2023]
    test_seasons = [2024, 2025]
    
    train_df = df[df['season'].isin(train_seasons)]
    val_df = df[df['season'].isin(val_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    print(f"\nTrain: {len(train_df)} games (seasons {min(train_seasons)}-{max(train_seasons)})")
    print(f"Val:   {len(val_df)} games (season {val_seasons})")
    print(f"Test:  {len(test_df)} games (seasons {test_seasons})")
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_win']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['home_win']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_win']
    
    # Train XGBoost
    print("\n" + "-" * 50)
    print("TRAINING: XGBoost V2")
    print("-" * 50)
    
    model, metrics, val_probs = train_xgboost_v2(X_train, y_train, X_val, y_val)
    
    print(f"Val Brier:    {metrics['val_brier']:.4f} (target: <{MARKET_BRIER:.4f})")
    print(f"Val LogLoss:  {metrics['val_logloss']:.4f} (target: <{MARKET_LOGLOSS:.4f})")
    print(f"Val Accuracy: {metrics['val_accuracy']:.1%}")
    
    # Calibrate
    print("\n" + "-" * 50)
    print("CALIBRATION: Isotonic Regression")
    print("-" * 50)
    
    calibrator, cal_metrics = calibrate_probabilities(model, X_val, y_val, val_probs)
    print(f"Calibrated Brier:   {cal_metrics['cal_brier']:.4f}")
    print(f"Calibrated LogLoss: {cal_metrics['cal_logloss']:.4f}")
    
    # Test set evaluation
    print("\n" + "-" * 50)
    print("TEST SET (Out-of-Sample 2024-2025)")
    print("-" * 50)
    
    test_probs_raw = model.predict_proba(X_test)[:, 1]
    test_probs = calibrator.predict(test_probs_raw)
    
    test_brier = brier_score_loss(y_test, test_probs)
    test_ll = log_loss(y_test, test_probs.clip(0.01, 0.99))
    test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))
    
    print(f"Test Brier:    {test_brier:.4f} (target: <{MARKET_BRIER:.4f})")
    print(f"Test LogLoss:  {test_ll:.4f} (target: <{MARKET_LOGLOSS:.4f})")
    print(f"Test Accuracy: {test_acc:.1%}")
    
    # Compare to market
    print("\n" + "-" * 50)
    print("VS MARKET BENCHMARK")
    print("-" * 50)
    
    market_comparison = evaluate_vs_market(test_df, y_test, test_probs, market_odds)
    
    if market_comparison:
        print(f"Games matched: {market_comparison['n_games']}")
        print(f"Model Brier:  {market_comparison['model_brier']:.4f}")
        print(f"Market Brier: {market_comparison['market_brier']:.4f}")
        print(f"Difference:   {market_comparison['brier_diff']:+.4f} ({'WORSE' if market_comparison['brier_diff'] > 0 else 'BETTER'})")
    
    # Feature importance
    print("\n" + "-" * 50)
    print("TOP 15 FEATURES")
    print("-" * 50)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_package = {
        'model': model,
        'calibrator': calibrator,
        'feature_cols': feature_cols,
        'train_seasons': train_seasons,
        'val_seasons': val_seasons,
        'test_seasons': test_seasons,
        'metrics': {
            'val': metrics,
            'cal': cal_metrics,
            'test': {
                'brier': test_brier,
                'logloss': test_ll,
                'accuracy': test_acc,
            },
            'market_comparison': market_comparison,
        },
        'benchmark': {
            'market_brier': MARKET_BRIER,
            'market_logloss': MARKET_LOGLOSS,
        },
        'feature_importance': importance.to_dict(),
    }
    
    model_path = MODEL_DIR / 'model_v2.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n{'='*70}")
    print(f"MODEL SAVED: {model_path}")
    
    # Summary
    beat_market = test_brier < MARKET_BRIER
    print(f"\n{'🎯 BEAT MARKET!' if beat_market else '❌ Below market - needs improvement'}")
    print(f"Gap: {test_brier - MARKET_BRIER:+.4f} Brier points")
    print(f"{'='*70}")
    
    return model_package


if __name__ == '__main__':
    main()

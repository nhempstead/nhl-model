#!/usr/bin/env python3
"""
Train NHL betting model with walk-forward validation.

Key principles:
1. Walk-forward: Train on past seasons, validate on next, test on held-out
2. No leakage: All features computed from pre-game data only
3. Calibration: Probabilities should mean what they say
4. Track what matters: CLV potential, not just accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models/trained')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """Load processed matchup data"""
    path = os.path.join(DATA_DIR, 'processed/matchups_featured.csv')
    df = pd.read_csv(path)
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    return df


def get_feature_cols(df):
    """Identify feature columns"""
    exclude = ['gameId', 'gameDate', 'season', 'h_team', 'a_team', 'home_win',
               'h_totalGoalsFor', 'h_totalGoalsAgainst', 'a_totalGoalsFor', 'a_totalGoalsAgainst']
    return [c for c in df.columns if c not in exclude]


def walk_forward_validation(df, train_seasons, val_seasons, test_seasons):
    """
    Proper walk-forward split:
    - Train on past seasons
    - Validate on next season(s)
    - Test on held-out final season(s)
    """
    train = df[df['season'].isin(train_seasons)]
    val = df[df['season'].isin(val_seasons)]
    test = df[df['season'].isin(test_seasons)]
    
    print(f"Train: {len(train)} games, seasons {min(train_seasons)}-{max(train_seasons)}")
    print(f"Val:   {len(val)} games, seasons {val_seasons}")
    print(f"Test:  {len(test)} games, seasons {test_seasons}")
    
    return train, val, test


def train_baseline(X_train, y_train, X_val, y_val):
    """Simple logistic regression baseline"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_scaled, y_train)
    
    train_pred = model.predict_proba(X_train_scaled)[:, 1]
    val_pred = model.predict_proba(X_val_scaled)[:, 1]
    
    metrics = {
        'train_brier': brier_score_loss(y_train, train_pred),
        'val_brier': brier_score_loss(y_val, val_pred),
        'val_accuracy': accuracy_score(y_val, (val_pred > 0.5).astype(int)),
        'val_logloss': log_loss(y_val, val_pred),
    }
    
    return model, scaler, metrics, val_pred


def train_xgboost(X_train, y_train, X_val, y_val):
    """XGBoost with calibration"""
    
    # Balanced parameters - enough capacity for confident predictions
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=10,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
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


def calibrate_model(model, X_val, y_val, val_probs):
    """Platt scaling calibration with home ice adjustment"""
    
    # Fit logistic regression on validation probabilities
    calibrator = LogisticRegression()
    calibrator.fit(val_probs.reshape(-1, 1), y_val)
    
    # Calibrated predictions
    cal_probs = calibrator.predict_proba(val_probs.reshape(-1, 1))[:, 1]
    
    # Home ice adjustment: actual home win rate is ~54% historically
    # Shift predictions toward home team by ~2-3%
    home_ice_boost = 0.025
    cal_probs_adjusted = cal_probs + home_ice_boost
    cal_probs_adjusted = np.clip(cal_probs_adjusted, 0.01, 0.99)
    
    cal_metrics = {
        'cal_brier': brier_score_loss(y_val, cal_probs_adjusted),
        'cal_logloss': log_loss(y_val, cal_probs_adjusted),
        'home_ice_boost': home_ice_boost,
    }
    
    return calibrator, cal_metrics


def evaluate_betting_potential(y_true, probs, threshold=0.03):
    """
    Evaluate potential for finding edges.
    
    Assumes market is ~52% implied for home team on average.
    Counts games where model disagrees significantly.
    """
    
    # Approximate market probability (home team ~52%)
    market_prob = 0.52
    
    # Games where model sees edge
    edge = probs - market_prob
    bet_home = edge > threshold
    bet_away = edge < -threshold
    
    # Outcomes
    home_wins = y_true.values if hasattr(y_true, 'values') else y_true
    
    home_roi = (home_wins[bet_home].mean() - 0.52) / 0.52 if bet_home.sum() > 0 else 0
    away_roi = ((1 - home_wins[bet_away]).mean() - 0.48) / 0.48 if bet_away.sum() > 0 else 0
    
    return {
        'bet_home_count': bet_home.sum(),
        'bet_away_count': bet_away.sum(),
        'total_bets': bet_home.sum() + bet_away.sum(),
        'bet_rate': (bet_home.sum() + bet_away.sum()) / len(probs),
        'home_roi_approx': home_roi,
        'away_roi_approx': away_roi,
    }


def main():
    print("="*70)
    print("NHL MODEL TRAINING - Walk-Forward Validation")
    print("="*70)
    
    # Load data
    df = load_data()
    feature_cols = get_feature_cols(df)
    
    print(f"\nData: {len(df)} games, {len(feature_cols)} features")
    print(f"Seasons: {sorted(df['season'].unique())}")
    
    # Walk-forward split
    # Train: 2008-2021 (14 seasons)
    # Val: 2022-2023 (2 seasons)
    # Test: 2024-2025 (2 seasons)
    train_seasons = list(range(2008, 2022))
    val_seasons = [2022, 2023]
    test_seasons = [2024, 2025]
    
    train_df, val_df, test_df = walk_forward_validation(
        df, train_seasons, val_seasons, test_seasons
    )
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_win']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['home_win']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_win']
    
    # Train baseline
    print("\n" + "-"*50)
    print("BASELINE: Logistic Regression")
    print("-"*50)
    lr_model, scaler, lr_metrics, lr_val_probs = train_baseline(X_train, y_train, X_val, y_val)
    print(f"Val Brier:    {lr_metrics['val_brier']:.4f}")
    print(f"Val Accuracy: {lr_metrics['val_accuracy']:.1%}")
    print(f"Val LogLoss:  {lr_metrics['val_logloss']:.4f}")
    
    # Train XGBoost
    print("\n" + "-"*50)
    print("MAIN MODEL: XGBoost")
    print("-"*50)
    xgb_model, xgb_metrics, xgb_val_probs = train_xgboost(X_train, y_train, X_val, y_val)
    print(f"Val Brier:    {xgb_metrics['val_brier']:.4f}")
    print(f"Val Accuracy: {xgb_metrics['val_accuracy']:.1%}")
    print(f"Val LogLoss:  {xgb_metrics['val_logloss']:.4f}")
    
    # Calibrate
    print("\n" + "-"*50)
    print("CALIBRATION: Platt Scaling")
    print("-"*50)
    calibrator, cal_metrics = calibrate_model(xgb_model, X_val, y_val, xgb_val_probs)
    print(f"Calibrated Brier:   {cal_metrics['cal_brier']:.4f}")
    print(f"Calibrated LogLoss: {cal_metrics['cal_logloss']:.4f}")
    
    # Test set evaluation
    print("\n" + "-"*50)
    print("TEST SET (Out-of-Sample)")
    print("-"*50)
    
    test_probs_raw = xgb_model.predict_proba(X_test)[:, 1]
    test_probs = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
    # Apply home ice adjustment
    home_ice_boost = cal_metrics.get('home_ice_boost', 0.025)
    test_probs = np.clip(test_probs + home_ice_boost, 0.01, 0.99)
    
    test_metrics = {
        'test_brier': brier_score_loss(y_test, test_probs),
        'test_accuracy': accuracy_score(y_test, (test_probs > 0.5).astype(int)),
        'test_logloss': log_loss(y_test, test_probs),
    }
    
    print(f"Test Brier:    {test_metrics['test_brier']:.4f}")
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.1%}")
    print(f"Test LogLoss:  {test_metrics['test_logloss']:.4f}")
    
    # Betting potential
    print("\n" + "-"*50)
    print("BETTING POTENTIAL (3% edge threshold)")
    print("-"*50)
    betting = evaluate_betting_potential(y_test, test_probs)
    print(f"Total bettable games: {betting['total_bets']} ({betting['bet_rate']:.1%} of games)")
    print(f"Home bets: {betting['bet_home_count']}, Away bets: {betting['bet_away_count']}")
    
    # Feature importance
    print("\n" + "-"*50)
    print("TOP 10 FEATURES")
    print("-"*50)
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_package = {
        'model': xgb_model,
        'calibrator': calibrator,
        'feature_cols': feature_cols,
        'train_seasons': train_seasons,
        'val_seasons': val_seasons,
        'test_seasons': test_seasons,
        'metrics': {
            'val': xgb_metrics,
            'cal': cal_metrics,
            'test': test_metrics,
        },
        'feature_importance': importance.to_dict(),
    }
    
    model_path = os.path.join(MODEL_DIR, 'model_v1.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n{'='*70}")
    print(f"MODEL SAVED: {model_path}")
    print(f"{'='*70}")
    
    return model_package


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Week 2: Ensemble Model Training

Build ensemble of diverse models:
1. XGBoost (gradient boosting)
2. LightGBM (faster, different splits)
3. CatBoost (handles categoricals)
4. Logistic Regression (linear baseline)
5. Ridge Classifier (regularized linear)

Stack with meta-learner for final predictions.
Target: Brier < 0.225 (match market)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'trained'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MARKET_BRIER = 0.2248


def load_data():
    """Load processed matchup data."""
    # Try advanced (most features) first
    advanced_path = DATA_DIR / 'processed' / 'matchups_with_advanced.csv'
    situational_path = DATA_DIR / 'processed' / 'matchups_with_situational.csv'
    
    if advanced_path.exists():
        df = pd.read_csv(advanced_path, low_memory=False)
        print(f"Loaded ADVANCED data: {len(df)} games")
    else:
        df = pd.read_csv(situational_path, low_memory=False)
        print(f"Loaded situational data: {len(df)} games")
    
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    return df


def get_feature_cols(df):
    """Get numeric feature columns."""
    exclude = ['gameId', 'gameDate', 'season', 'h_team', 'a_team', 'home_win',
               'h_totalGoalsFor', 'h_totalGoalsAgainst', 'a_totalGoalsFor', 'a_totalGoalsAgainst',
               'home_goalie_id', 'away_goalie_id', 'home_goalie_name', 'away_goalie_name',
               'h_division', 'a_division', 'date', 'market_home_prob']
    
    numeric_cols = df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    return [c for c in numeric_cols if c not in exclude and not c.startswith('Unnamed')]


def train_base_models(X_train, y_train, X_val, y_val):
    """Train diverse base models."""
    
    models = {}
    val_preds = {}
    
    # 1. XGBoost
    print("  Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=20,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['xgb'] = xgb
    val_preds['xgb'] = xgb.predict_proba(X_val)[:, 1]
    
    # 2. LightGBM
    print("  Training LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        min_child_weight=20,
        random_state=42,
        verbose=-1,
    )
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    models['lgb'] = lgb
    val_preds['lgb'] = lgb.predict_proba(X_val)[:, 1]
    
    # 3. Logistic Regression (scaled)
    print("  Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lr = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
    )
    lr.fit(X_train_scaled, y_train)
    models['lr'] = lr
    models['scaler'] = scaler
    val_preds['lr'] = lr.predict_proba(X_val_scaled)[:, 1]
    
    # 4. Regularized LogReg (different C)
    print("  Training Ridge LogReg...")
    lr2 = LogisticRegression(
        C=0.01,
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
    )
    lr2.fit(X_train_scaled, y_train)
    models['lr_ridge'] = lr2
    val_preds['lr_ridge'] = lr2.predict_proba(X_val_scaled)[:, 1]
    
    return models, val_preds


def train_meta_learner(val_preds, y_val):
    """Train meta-learner on base model predictions."""
    
    # Stack predictions as features
    meta_X = np.column_stack([val_preds[m] for m in ['xgb', 'lgb', 'lr', 'lr_ridge']])
    
    # Simple averaging baseline
    avg_pred = meta_X.mean(axis=1)
    avg_brier = brier_score_loss(y_val, avg_pred)
    print(f"\n  Simple average Brier: {avg_brier:.4f}")
    
    # Weighted average (optimize weights)
    print("  Optimizing ensemble weights...")
    best_weights = None
    best_brier = float('inf')
    
    # Grid search over weight combinations
    for w1 in np.arange(0.1, 0.6, 0.1):
        for w2 in np.arange(0.1, 0.6, 0.1):
            for w3 in np.arange(0.0, 0.4, 0.1):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0 or w4 > 0.5:
                    continue
                
                weights = np.array([w1, w2, w3, w4])
                weighted_pred = (meta_X * weights).sum(axis=1)
                brier = brier_score_loss(y_val, weighted_pred)
                
                if brier < best_brier:
                    best_brier = brier
                    best_weights = weights
    
    print(f"  Best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, LR={best_weights[2]:.2f}, Ridge={best_weights[3]:.2f}")
    print(f"  Weighted ensemble Brier: {best_brier:.4f}")
    
    # Logistic meta-learner
    meta_lr = LogisticRegression(C=1.0, max_iter=1000)
    meta_lr.fit(meta_X, y_val)
    stacked_pred = meta_lr.predict_proba(meta_X)[:, 1]
    stacked_brier = brier_score_loss(y_val, stacked_pred)
    print(f"  Stacked meta-learner Brier: {stacked_brier:.4f}")
    
    # Use best approach
    if best_brier <= stacked_brier and best_brier <= avg_brier:
        return {'type': 'weighted', 'weights': best_weights}, best_brier
    elif stacked_brier <= avg_brier:
        return {'type': 'stacked', 'model': meta_lr}, stacked_brier
    else:
        return {'type': 'average'}, avg_brier


def evaluate_ensemble(models, meta, X, y, scaler):
    """Evaluate ensemble on data."""
    
    # Get base predictions
    preds = {}
    preds['xgb'] = models['xgb'].predict_proba(X)[:, 1]
    preds['lgb'] = models['lgb'].predict_proba(X)[:, 1]
    
    X_scaled = scaler.transform(X)
    preds['lr'] = models['lr'].predict_proba(X_scaled)[:, 1]
    preds['lr_ridge'] = models['lr_ridge'].predict_proba(X_scaled)[:, 1]
    
    # Combine using meta-learner
    meta_X = np.column_stack([preds[m] for m in ['xgb', 'lgb', 'lr', 'lr_ridge']])
    
    if meta['type'] == 'weighted':
        ensemble_pred = (meta_X * meta['weights']).sum(axis=1)
    elif meta['type'] == 'stacked':
        ensemble_pred = meta['model'].predict_proba(meta_X)[:, 1]
    else:
        ensemble_pred = meta_X.mean(axis=1)
    
    # Metrics
    brier = brier_score_loss(y, ensemble_pred)
    ll = log_loss(y, ensemble_pred.clip(0.01, 0.99))
    acc = accuracy_score(y, (ensemble_pred > 0.5).astype(int))
    
    return ensemble_pred, {'brier': brier, 'logloss': ll, 'accuracy': acc}


def main():
    print("=" * 70)
    print("ENSEMBLE MODEL TRAINING - WEEK 2")
    print(f"Target: Beat market Brier {MARKET_BRIER:.4f}")
    print("=" * 70)
    
    # Load data
    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"Features: {len(feature_cols)}")
    
    # Split data
    train_seasons = list(range(2008, 2023))
    val_seasons = [2023]
    test_seasons = [2024, 2025]
    
    train_df = df[df['season'].isin(train_seasons)]
    val_df = df[df['season'].isin(val_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['home_win']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['home_win']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['home_win']
    
    # Train base models
    print("\n" + "-" * 50)
    print("TRAINING BASE MODELS")
    print("-" * 50)
    
    models, val_preds = train_base_models(X_train, y_train, X_val, y_val)
    
    # Show individual model performance
    print("\nBase model validation performance:")
    for name, pred in val_preds.items():
        brier = brier_score_loss(y_val, pred)
        print(f"  {name}: Brier={brier:.4f}")
    
    # Train meta-learner
    print("\n" + "-" * 50)
    print("TRAINING META-LEARNER")
    print("-" * 50)
    
    meta, meta_val_brier = train_meta_learner(val_preds, y_val)
    
    # Calibrate ensemble
    print("\n" + "-" * 50)
    print("CALIBRATION")
    print("-" * 50)
    
    val_ensemble_pred, val_metrics = evaluate_ensemble(models, meta, X_val, y_val, models['scaler'])
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_ensemble_pred, y_val)
    
    cal_pred = calibrator.predict(val_ensemble_pred)
    cal_brier = brier_score_loss(y_val, cal_pred)
    print(f"Pre-calibration Brier:  {val_metrics['brier']:.4f}")
    print(f"Post-calibration Brier: {cal_brier:.4f}")
    
    # Test set evaluation
    print("\n" + "-" * 50)
    print("TEST SET (2024-2025)")
    print("-" * 50)
    
    test_ensemble_pred, test_metrics = evaluate_ensemble(models, meta, X_test, y_test, models['scaler'])
    test_cal_pred = calibrator.predict(test_ensemble_pred)
    
    test_brier = brier_score_loss(y_test, test_cal_pred)
    test_ll = log_loss(y_test, test_cal_pred.clip(0.01, 0.99))
    test_acc = accuracy_score(y_test, (test_cal_pred > 0.5).astype(int))
    
    print(f"Ensemble Brier:    {test_brier:.4f} (target: <{MARKET_BRIER:.4f})")
    print(f"Ensemble LogLoss:  {test_ll:.4f}")
    print(f"Ensemble Accuracy: {test_acc:.1%}")
    
    # Compare to single XGBoost
    xgb_test_pred = models['xgb'].predict_proba(X_test)[:, 1]
    xgb_brier = brier_score_loss(y_test, xgb_test_pred)
    print(f"\nSingle XGBoost Brier: {xgb_brier:.4f}")
    print(f"Ensemble improvement: {xgb_brier - test_brier:+.4f}")
    
    # Save model
    ensemble_package = {
        'models': models,
        'meta': meta,
        'calibrator': calibrator,
        'feature_cols': feature_cols,
        'metrics': {
            'val_brier': cal_brier,
            'test_brier': test_brier,
            'test_logloss': test_ll,
        }
    }
    
    model_path = MODEL_DIR / 'ensemble_v1.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_package, f)
    
    print(f"\n{'='*70}")
    print(f"MODEL SAVED: {model_path}")
    
    gap = test_brier - MARKET_BRIER
    if gap < 0:
        print(f"\n🎯 BEAT MARKET by {-gap:.4f} Brier points!")
    else:
        print(f"\n❌ Still {gap:.4f} behind market")
    print(f"{'='*70}")
    
    return ensemble_package


if __name__ == '__main__':
    main()

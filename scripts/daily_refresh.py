#!/usr/bin/env python3
"""
NHL Model Daily Data Refresh

Comprehensive daily update of all data sources:
1. MoneyPuck team stats (current season xG, Corsi, etc.)
2. MoneyPuck goalie stats (current season save %, xG against)
3. NHL official rosters
4. Daily Faceoff injuries
5. ESPN transactions
6. Power ratings calculation

Run every morning before predictions to ensure model uses fresh data.

Usage:
    python daily_refresh.py              # Full refresh
    python daily_refresh.py --quick      # Just injuries + transactions
    python daily_refresh.py --teams NYR LAK  # Specific teams only
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from io import StringIO
import requests
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
RAW_DIR = DATA_DIR / 'raw'
RATINGS_DIR = DATA_DIR / 'ratings'

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
RATINGS_DIR.mkdir(parents=True, exist_ok=True)

# Current NHL season
CURRENT_SEASON = '2025'


def fetch_moneypuck_teams() -> pd.DataFrame:
    """Fetch current season team stats from MoneyPuck."""
    print("  Fetching MoneyPuck team stats...", end=' ', flush=True)
    
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{CURRENT_SEASON}/regular/teams.csv"
    
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            df['fetched_at'] = datetime.now(timezone.utc).isoformat()
            
            # Save to file
            output_path = RAW_DIR / 'moneypuck_teams_current.csv'
            df.to_csv(output_path, index=False)
            
            # Count unique teams
            teams = df['team'].nunique() if 'team' in df.columns else len(df)
            print(f"✓ {teams} teams, {len(df)} rows")
            return df
        else:
            print(f"✗ HTTP {resp.status_code}")
    except Exception as e:
        print(f"✗ {e}")
    
    return pd.DataFrame()


def fetch_moneypuck_goalies() -> pd.DataFrame:
    """Fetch current season goalie stats from MoneyPuck."""
    print("  Fetching MoneyPuck goalie stats...", end=' ', flush=True)
    
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{CURRENT_SEASON}/regular/goalies.csv"
    
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            df['fetched_at'] = datetime.now(timezone.utc).isoformat()
            
            # Save to file
            output_path = RAW_DIR / 'moneypuck_goalies_current.csv'
            df.to_csv(output_path, index=False)
            
            goalies = df['name'].nunique() if 'name' in df.columns else len(df)
            print(f"✓ {goalies} goalies")
            return df
        else:
            print(f"✗ HTTP {resp.status_code}")
    except Exception as e:
        print(f"✗ {e}")
    
    return pd.DataFrame()


def fetch_moneypuck_games() -> pd.DataFrame:
    """Fetch current season game-by-game data from MoneyPuck."""
    print("  Fetching MoneyPuck game data...", end=' ', flush=True)
    
    # Try different URL patterns
    urls = [
        f"https://moneypuck.com/moneypuck/playerData/games/{CURRENT_SEASON}/regular/games.csv",
        f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{CURRENT_SEASON}/regular/games.csv",
    ]
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                df['fetched_at'] = datetime.now(timezone.utc).isoformat()
                
                # Save to file
                output_path = RAW_DIR / 'moneypuck_games_current.csv'
                df.to_csv(output_path, index=False)
                
                games = df['game_id'].nunique() if 'game_id' in df.columns else len(df) // 2
                print(f"✓ {games} games")
                return df
        except:
            continue
    
    print("✗ Failed")
    return pd.DataFrame()


def run_roster_refresh(teams: list = None) -> bool:
    """Run NHL roster refresh script."""
    print("  Refreshing NHL rosters...", end=' ', flush=True)
    
    script = SCRIPT_DIR / 'collect' / 'nhl_rosters.py'
    cmd = ['python', str(script)]
    if teams:
        cmd.extend(teams)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Count from output
            lines = result.stdout.strip().split('\n')
            summary = [l for l in lines if 'Fetched' in l]
            if summary:
                print(f"✓ {summary[-1].split('✓')[-1].strip()}")
            else:
                print("✓")
            return True
    except Exception as e:
        print(f"✗ {e}")
    
    return False


def run_injury_refresh(teams: list = None) -> bool:
    """Run Daily Faceoff injury refresh script."""
    print("  Refreshing injury data...", end=' ', flush=True)
    
    script = SCRIPT_DIR / 'collect' / 'daily_faceoff_rosters.py'
    cmd = ['python', str(script)]
    if teams:
        cmd.extend(teams)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Count injuries from output
            injury_lines = [l for l in result.stdout.split('\n') if 'GOALIE' in l]
            if injury_lines:
                print(f"✓ {len(injury_lines)} goalie injuries detected")
            else:
                print("✓ No critical injuries")
            return True
    except Exception as e:
        print(f"✗ {e}")
    
    return False


def run_transaction_check(teams: list = None) -> dict:
    """Run ESPN transaction check."""
    print("  Checking transactions...", end=' ', flush=True)
    
    script = SCRIPT_DIR / 'collect' / 'nhl_transactions.py'
    cmd = ['python', str(script), '--days', '3', '--json']
    if teams:
        cmd.extend(['--teams'] + teams)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            trades = [t for t in data.get('transactions', []) if t.get('type') == 'TRADE']
            print(f"✓ {len(trades)} recent trades")
            return data
    except Exception as e:
        print(f"✗ {e}")
    
    return {}


def calculate_power_ratings(team_stats: pd.DataFrame, goalie_stats: pd.DataFrame) -> dict:
    """
    Calculate power ratings from current stats.
    
    Power rating components:
    - Offensive rating: xGF/GP (expected goals for per game)
    - Defensive rating: xGA/GP (expected goals against per game)
    - Percentages: Corsi%, xG%
    - Luck factor: actual goals vs expected
    
    Rating scale: 100 = league average, higher = better
    """
    print("  Calculating power ratings...", end=' ', flush=True)
    
    if team_stats.empty:
        print("✗ No team stats")
        return {}
    
    ratings = {}
    
    # Filter to 5on5 situation for core metrics
    if 'situation' in team_stats.columns:
        core = team_stats[team_stats['situation'] == '5on5'].copy()
    else:
        core = team_stats.copy()
    
    # Get unique teams
    if 'team' not in core.columns:
        print("✗ No team column")
        return {}
    
    # Calculate league averages first
    all_xgf_per_game = []
    all_xga_per_game = []
    
    for team in core['team'].unique():
        team_data = core[core['team'] == team]
        if len(team_data) == 0:
            continue
        row = team_data.iloc[0]
        gp = row.get('games_played', 1) or 1
        xgf = row.get('xGoalsFor', 0) or 0
        xga = row.get('xGoalsAgainst', 0) or 0
        all_xgf_per_game.append(xgf / gp)
        all_xga_per_game.append(xga / gp)
    
    league_avg_xgf = sum(all_xgf_per_game) / len(all_xgf_per_game) if all_xgf_per_game else 2.5
    league_avg_xga = sum(all_xga_per_game) / len(all_xga_per_game) if all_xga_per_game else 2.5
    
    for team in core['team'].unique():
        team_data = core[core['team'] == team]
        if len(team_data) == 0:
            continue
        row = team_data.iloc[0]
        
        # Get games played for per-game calculations
        gp = row.get('games_played', 1) or 1
        
        # Extract cumulative metrics and convert to per-game
        xgf_total = row.get('xGoalsFor', 0) or 0
        xga_total = row.get('xGoalsAgainst', 0) or 0
        gf_total = row.get('goalsFor', 0) or 0
        ga_total = row.get('goalsAgainst', 0) or 0
        
        xgf_per_game = xgf_total / gp
        xga_per_game = xga_total / gp
        gf_per_game = gf_total / gp
        ga_per_game = ga_total / gp
        
        # Percentages are already normalized (0-1 scale in MoneyPuck)
        cf_pct = row.get('corsiPercentage', 0.50) or 0.50
        xg_pct = row.get('xGoalsPercentage', 0.50) or 0.50
        
        # Convert to 0-100 scale if needed
        if cf_pct <= 1:
            cf_pct *= 100
        if xg_pct <= 1:
            xg_pct *= 100
        
        # Calculate ratings (100 = league average)
        # Offense: higher xGF/game = better
        off_rating = 100 + (xgf_per_game - league_avg_xgf) * 20
        
        # Defense: lower xGA/game = better  
        def_rating = 100 - (xga_per_game - league_avg_xga) * 20
        
        # Overall power rating
        power = (off_rating + def_rating) / 2
        
        # Luck factor (actual vs expected) - >1 means outperforming xG
        if xgf_total > 0 and xga_total > 0 and ga_total > 0:
            goal_luck = gf_total / xgf_total  # >1 = scoring more than expected
            save_luck = xga_total / ga_total   # >1 = allowing less than expected
            luck = (goal_luck + save_luck) / 2
        else:
            luck = 1.0
        
        ratings[team] = {
            'power_rating': round(float(power), 1),
            'off_rating': round(float(off_rating), 1),
            'def_rating': round(float(def_rating), 1),
            'xGF_per_game': round(float(xgf_per_game), 2),
            'xGA_per_game': round(float(xga_per_game), 2),
            'GF_per_game': round(float(gf_per_game), 2),
            'GA_per_game': round(float(ga_per_game), 2),
            'games_played': int(gp),
            'corsi_pct': round(float(cf_pct), 1),
            'xG_pct': round(float(xg_pct), 1),
            'luck_factor': round(float(luck), 3),
        }
    
    # Add goalie ratings if available
    if not goalie_stats.empty and 'team' in goalie_stats.columns:
        for team in ratings:
            team_goalies = goalie_stats[goalie_stats['team'] == team]
            if len(team_goalies) > 0:
                # Get primary goalie (most games)
                if 'games_played' in team_goalies.columns:
                    primary = team_goalies.sort_values('games_played', ascending=False).iloc[0]
                else:
                    primary = team_goalies.iloc[0]
                
                gsax = primary.get('xGoals_saved_above_expected', primary.get('GSAx', 0))
                sv_pct = primary.get('save_percentage', primary.get('SV%', 0.900))
                
                ratings[team]['goalie_gsax'] = round(gsax, 2) if pd.notna(gsax) else 0
                ratings[team]['goalie_sv_pct'] = round(sv_pct, 4) if pd.notna(sv_pct) else 0.900
    
    # Save ratings
    output_path = RATINGS_DIR / 'power_ratings.json'
    with open(output_path, 'w') as f:
        json.dump({
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'season': CURRENT_SEASON,
            'ratings': ratings
        }, f, indent=2)
    
    print(f"✓ {len(ratings)} teams rated")
    return ratings


def print_top_ratings(ratings: dict, n: int = 10):
    """Print top N teams by power rating."""
    if not ratings:
        return
    
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1]['power_rating'], reverse=True)
    
    print(f"\n{'='*60}")
    print("TOP POWER RATINGS (100 = league avg)")
    print('='*60)
    print(f"{'Rank':<5} {'Team':<5} {'Power':<7} {'Off':<7} {'Def':<7} {'xGF/G':<7} {'xGA/G':<7} {'Luck':<6}")
    print('-'*60)
    
    for i, (team, r) in enumerate(sorted_teams[:n], 1):
        print(f"{i:<5} {team:<5} {r['power_rating']:<7.1f} {r['off_rating']:<7.1f} {r['def_rating']:<7.1f} {r['xGF_per_game']:<7.2f} {r['xGA_per_game']:<7.2f} {r['luck_factor']:<6.3f}")


def main():
    parser = argparse.ArgumentParser(description='NHL Model Daily Refresh')
    parser.add_argument('--quick', action='store_true', help='Quick refresh (injuries + transactions only)')
    parser.add_argument('--teams', nargs='*', help='Specific teams to refresh')
    parser.add_argument('--no-ratings', action='store_true', help='Skip power ratings calculation')
    args = parser.parse_args()
    
    print("="*60)
    print(f"NHL MODEL DAILY REFRESH - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    teams = [t.upper() for t in args.teams] if args.teams else None
    
    team_stats = pd.DataFrame()
    goalie_stats = pd.DataFrame()
    
    if not args.quick:
        print("\n📊 STATS REFRESH")
        team_stats = fetch_moneypuck_teams()
        goalie_stats = fetch_moneypuck_goalies()
        fetch_moneypuck_games()
    
    print("\n📋 ROSTER REFRESH")
    run_roster_refresh(teams)
    run_injury_refresh(teams)
    
    print("\n📰 TRANSACTION CHECK")
    run_transaction_check(teams)
    
    if not args.quick and not args.no_ratings:
        print("\n⚡ POWER RATINGS")
        ratings = calculate_power_ratings(team_stats, goalie_stats)
        print_top_ratings(ratings)
    
    print("\n" + "="*60)
    print("✅ DAILY REFRESH COMPLETE")
    print("="*60)
    
    # Summary
    print(f"\nData saved to: {DATA_DIR}")
    print("Run predictions: python scripts/predict/today.py")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Collect game-level xG data from MoneyPuck.
This gives us actual expected goals per game, not season aggregates.
"""

import requests
import pandas as pd
import os
from io import StringIO

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
os.makedirs(DATA_DIR, exist_ok=True)

def collect_team_game_stats():
    """
    MoneyPuck provides game-by-game team stats.
    URL pattern: moneypuck.com/moneypuck/playerData/games/{season}/regular/teams/games.csv
    """
    
    seasons = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
    
    all_data = []
    
    for season in seasons:
        print(f"Fetching {season}...")
        
        # MoneyPuck URL for team game-by-game data
        url = f"https://moneypuck.com/moneypuck/playerData/games/{season}/regular/games.csv"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                df['season'] = season
                all_data.append(df)
                print(f"  ✓ {len(df)} game records")
            else:
                print(f"  ✗ Status {resp.status_code}")
                
                # Try alternative URL structure
                alt_url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/games.csv"
                resp = requests.get(alt_url, timeout=30)
                if resp.status_code == 200:
                    df = pd.read_csv(StringIO(resp.text))
                    df['season'] = season
                    all_data.append(df)
                    print(f"  ✓ (alt) {len(df)} game records")
                else:
                    print(f"  ✗ (alt) Status {resp.status_code}")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(DATA_DIR, 'moneypuck_games.csv')
        combined.to_csv(output_path, index=False)
        print(f"\nSaved {len(combined)} rows to {output_path}")
        return combined
    
    return pd.DataFrame()


def collect_goalie_game_stats():
    """
    Collect goalie game-by-game stats from MoneyPuck.
    """
    
    seasons = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
    
    all_data = []
    
    for season in seasons:
        print(f"Fetching goalies {season}...")
        
        # Try different URL patterns
        urls = [
            f"https://moneypuck.com/moneypuck/playerData/games/{season}/regular/goalies.csv",
            f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/goalies.csv",
        ]
        
        for url in urls:
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    df = pd.read_csv(StringIO(resp.text))
                    df['season'] = season
                    all_data.append(df)
                    print(f"  ✓ {len(df)} goalie records from {url.split('/')[-1]}")
                    break
            except Exception as e:
                continue
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(DATA_DIR, 'moneypuck_goalies.csv')
        combined.to_csv(output_path, index=False)
        print(f"\nSaved {len(combined)} rows to {output_path}")
        return combined
    
    return pd.DataFrame()


def collect_team_season_stats():
    """
    Collect team season summary stats (for baseline power ratings).
    """
    
    seasons = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
    
    all_data = []
    
    for season in seasons:
        print(f"Fetching team stats {season}...")
        
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                df['season'] = season
                all_data.append(df)
                print(f"  ✓ {len(df)} team-season records")
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(DATA_DIR, 'moneypuck_teams.csv')
        combined.to_csv(output_path, index=False)
        print(f"\nSaved {len(combined)} rows to {output_path}")
        return combined
    
    return pd.DataFrame()


if __name__ == '__main__':
    print("="*60)
    print("MoneyPuck Data Collector")
    print("="*60)
    
    print("\n--- Team Game Stats ---")
    games = collect_team_game_stats()
    
    print("\n--- Goalie Stats ---")
    goalies = collect_goalie_game_stats()
    
    print("\n--- Team Season Stats ---")
    teams = collect_team_season_stats()
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)

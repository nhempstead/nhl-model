#!/usr/bin/env python3
"""
Collect NHL boxscore data for recent seasons.
Extracts player-level and team-level game stats.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
OUTPUT_DIR = BASE_DIR / 'data' / 'boxscores'
OUTPUT_DIR.mkdir(exist_ok=True)

API_BASE = "https://api-web.nhle.com/v1"


def get_season_games(season_start_year):
    """Get all game IDs for a season."""
    games = []
    
    # Regular season games are 0001-1312 approximately
    for game_num in range(1, 1400):
        game_id = f"{season_start_year}02{game_num:04d}"
        games.append(game_id)
    
    return games


def fetch_boxscore(game_id):
    """Fetch boxscore for a single game."""
    url = f"{API_BASE}/gamecenter/{game_id}/boxscore"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None  # Game doesn't exist
    except Exception as e:
        print(f"Error fetching {game_id}: {e}")
    
    return None


def extract_team_stats(boxscore):
    """Extract team-level stats from boxscore."""
    if not boxscore:
        return None
    
    try:
        game_id = boxscore.get('id')
        
        home = boxscore.get('homeTeam', {})
        away = boxscore.get('awayTeam', {})
        
        stats = {
            'game_id': game_id,
            'home_team': home.get('abbrev'),
            'away_team': away.get('abbrev'),
            'home_score': home.get('score'),
            'away_score': away.get('score'),
        }
        
        # Get team game stats
        team_stats = boxscore.get('teamGameStats', [])
        for ts in team_stats:
            cat = ts.get('category', '')
            home_val = ts.get('homeValue')
            away_val = ts.get('awayValue')
            
            stats[f'home_{cat}'] = home_val
            stats[f'away_{cat}'] = away_val
        
        return stats
    except Exception as e:
        print(f"Error extracting: {e}")
        return None


def collect_season(season_year, max_games=100):
    """Collect boxscores for a season."""
    print(f"\nCollecting {season_year}-{season_year+1} season...")
    
    results = []
    errors = 0
    
    for game_num in range(1, max_games + 1):
        game_id = f"{season_year}02{game_num:04d}"
        
        box = fetch_boxscore(game_id)
        if box:
            stats = extract_team_stats(box)
            if stats:
                results.append(stats)
        else:
            errors += 1
            if errors > 20:  # Stop if too many missing
                break
        
        if game_num % 50 == 0:
            print(f"  {game_num} games processed, {len(results)} collected")
        
        time.sleep(0.1)  # Rate limit
    
    return results


def main():
    print("="*60)
    print("NHL BOXSCORE COLLECTION")
    print("="*60)
    
    # Collect recent seasons
    all_stats = []
    
    for year in [2023, 2024]:
        stats = collect_season(year, max_games=200)
        all_stats.extend(stats)
        print(f"  Season {year}: {len(stats)} games")
    
    # Save
    df = pd.DataFrame(all_stats)
    output_file = OUTPUT_DIR / 'nhl_boxscores.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved {len(df)} games to {output_file}")
    print(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == '__main__':
    main()

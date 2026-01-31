#!/usr/bin/env python3
"""
Collect historical NHL game data from NHL API.
Target: All regular season games from 2015-2024 (~10,000 games)
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw')
os.makedirs(DATA_DIR, exist_ok=True)

def get_season_games(season: str) -> list:
    """
    Fetch all regular season games for a given season.
    Season format: '20232024' for the 2023-24 season
    """
    url = f"https://api-web.nhle.com/v1/schedule/{season[:4]}-10-01"
    
    all_games = []
    
    # NHL API returns schedule week by week, need to paginate
    # Alternative: use the season schedule endpoint
    schedule_url = f"https://api-web.nhle.com/v1/club-schedule-season/TOR/{season}"
    
    # Actually, let's use the game-by-game approach via standings/scores
    # The NHL API has changed - let's try the scores endpoint
    
    # For historical data, we'll iterate through dates
    start_year = int(season[:4])
    
    # Regular season roughly Oct 1 - Apr 15
    from datetime import date, timedelta
    
    start_date = date(start_year, 10, 1)
    end_date = date(start_year + 1, 4, 30)
    
    current = start_date
    games_collected = 0
    
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        url = f"https://api-web.nhle.com/v1/score/{date_str}"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                games = data.get('games', [])
                
                for game in games:
                    if game.get('gameType') == 2:  # Regular season
                        game_data = {
                            'game_id': game.get('id'),
                            'date': date_str,
                            'season': season,
                            'home_team': game.get('homeTeam', {}).get('abbrev'),
                            'away_team': game.get('awayTeam', {}).get('abbrev'),
                            'home_score': game.get('homeTeam', {}).get('score'),
                            'away_score': game.get('awayTeam', {}).get('score'),
                            'game_state': game.get('gameState'),
                            'period': game.get('period'),
                            'venue': game.get('venue', {}).get('default'),
                        }
                        
                        # Add goalie info if available
                        if 'homeTeam' in game:
                            ht = game['homeTeam']
                            if 'goalie' in ht:
                                game_data['home_goalie_id'] = ht['goalie'].get('id')
                                game_data['home_goalie_name'] = ht['goalie'].get('name', {}).get('default')
                        
                        if 'awayTeam' in game:
                            at = game['awayTeam']
                            if 'goalie' in at:
                                game_data['away_goalie_id'] = at['goalie'].get('id')
                                game_data['away_goalie_name'] = at['goalie'].get('name', {}).get('default')
                        
                        all_games.append(game_data)
                        games_collected += 1
                
        except Exception as e:
            print(f"  Error on {date_str}: {e}")
        
        current += timedelta(days=1)
        
        # Progress every 30 days
        if (current - start_date).days % 30 == 0:
            print(f"  {season}: {date_str} - {games_collected} games")
        
        # Small delay to be nice to API
        time.sleep(0.05)
    
    return all_games


def collect_all_seasons():
    """Collect data for all seasons 2015-2024"""
    
    seasons = [
        '20152016', '20162017', '20172018', '20182019', '20192020',
        '20202021', '20212022', '20222023', '20232024', '20242025'
    ]
    
    all_data = []
    
    for season in seasons:
        print(f"\nCollecting {season}...")
        games = get_season_games(season)
        all_data.extend(games)
        print(f"  Total: {len(games)} games")
        
        # Save incrementally
        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(DATA_DIR, 'nhl_games_historical.csv'), index=False)
        print(f"  Saved. Total rows: {len(df)}")
    
    return pd.DataFrame(all_data)


if __name__ == '__main__':
    print("="*60)
    print("NHL Historical Games Collector")
    print("="*60)
    
    df = collect_all_seasons()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(df)} games collected")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Seasons: {df['season'].nunique()}")
    print(f"Saved to: {DATA_DIR}/nhl_games_historical.csv")

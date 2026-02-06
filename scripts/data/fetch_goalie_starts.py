#!/usr/bin/env python3
"""
Fetch historical starting goalie data.

Sources:
1. NHL API game data (has actual starters)
2. Daily Faceoff archives (projected starters)

This gives us game-level goalie features instead of season averages.
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
GOALIE_DIR = DATA_DIR / 'goalie_starts'
GOALIE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_nhl_game(game_id: int) -> dict:
    """Fetch game data from NHL API."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None


def extract_starting_goalies(game_data: dict) -> dict:
    """Extract starting goalies from game boxscore."""
    if not game_data:
        return None
    
    try:
        home_team = game_data.get('homeTeam', {})
        away_team = game_data.get('awayTeam', {})
        
        # Find goalies who played
        home_goalies = []
        away_goalies = []
        
        for player in game_data.get('playerByGameStats', {}).get('homeTeam', {}).get('goalies', []):
            if player.get('toi', '0:00') != '0:00':
                home_goalies.append({
                    'name': f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}",
                    'toi': player.get('toi'),
                    'saves': player.get('saves', 0),
                    'goals_against': player.get('goalsAgainst', 0),
                })
        
        for player in game_data.get('playerByGameStats', {}).get('awayTeam', {}).get('goalies', []):
            if player.get('toi', '0:00') != '0:00':
                away_goalies.append({
                    'name': f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}",
                    'toi': player.get('toi'),
                    'saves': player.get('saves', 0),
                    'goals_against': player.get('goalsAgainst', 0),
                })
        
        # Starter is the one with most TOI
        home_starter = max(home_goalies, key=lambda x: x['toi']) if home_goalies else None
        away_starter = max(away_goalies, key=lambda x: x['toi']) if away_goalies else None
        
        return {
            'home_team': home_team.get('abbrev'),
            'away_team': away_team.get('abbrev'),
            'home_starter': home_starter['name'] if home_starter else None,
            'away_starter': away_starter['name'] if away_starter else None,
            'home_starter_saves': home_starter['saves'] if home_starter else None,
            'away_starter_saves': away_starter['saves'] if away_starter else None,
        }
    except Exception as e:
        print(f"Error extracting goalies: {e}")
        return None


def fetch_season_schedule(season: int) -> list:
    """Fetch all games for a season."""
    # Season format: 20232024 for 2023-24 season
    season_str = f"{season}{season+1}"
    url = f"https://api-web.nhle.com/v1/schedule/{season_str}"
    
    try:
        # Fetch week by week
        games = []
        start_date = datetime(season, 10, 1)
        end_date = datetime(season + 1, 6, 30)
        
        current = start_date
        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            day_url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
            
            try:
                resp = requests.get(day_url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for week in data.get('gameWeek', []):
                        for game in week.get('games', []):
                            if game.get('gameType') == 2:  # Regular season
                                games.append({
                                    'game_id': game['id'],
                                    'date': game['gameDate'],
                                    'home_team': game['homeTeam']['abbrev'],
                                    'away_team': game['awayTeam']['abbrev'],
                                })
            except:
                pass
            
            current += timedelta(days=7)
            time.sleep(0.1)
        
        return games
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []


def fetch_goalie_starts_for_season(season: int) -> pd.DataFrame:
    """Fetch all starting goalies for a season."""
    print(f"\n{'='*60}")
    print(f"Fetching goalie starts for {season}-{season+1} season")
    print(f"{'='*60}")
    
    # Try to load from our existing odds data to get game list
    odds_file = DATA_DIR / 'historical_odds_api' / f'nhl_odds_{season}-{season+1}.csv'
    
    if odds_file.exists():
        odds = pd.read_csv(odds_file)
        print(f"Found {len(odds)} games from odds data")
        
        # Extract game IDs from commence_time + teams
        # We'll need to match with NHL API
        games = []
        for _, row in odds.iterrows():
            games.append({
                'date': row['commence_time'][:10],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
            })
    else:
        print("No odds file, fetching schedule from NHL API...")
        games = fetch_season_schedule(season)
    
    if not games:
        print("No games found")
        return pd.DataFrame()
    
    print(f"Processing {len(games)} games...")
    
    # For now, return placeholder - full implementation would fetch each game
    # This is time-intensive, so we'll batch it
    
    return pd.DataFrame(games)


def main():
    """Fetch goalie starts for all seasons with odds data."""
    seasons = [2020, 2021, 2022, 2023, 2024]
    
    all_data = []
    for season in seasons:
        df = fetch_goalie_starts_for_season(season)
        if not df.empty:
            df['season'] = season
            all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_path = GOALIE_DIR / 'goalie_starts_all.csv'
        combined.to_csv(output_path, index=False)
        print(f"\nSaved {len(combined)} games to {output_path}")


if __name__ == '__main__':
    main()

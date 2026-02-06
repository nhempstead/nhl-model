#!/usr/bin/env python3
"""
Fetch historical NHL odds from The Odds API.

Endpoint: GET /v4/historical/sports/{sport}/odds
Parameters:
  - apiKey: Your API key
  - date: ISO 8601 timestamp for the snapshot
  - regions: us (includes Pinnacle, FanDuel, DraftKings, etc.)
  - markets: h2h (moneyline)
  - oddsFormat: american

Cost: 10 credits per region per market per request
Data available from June 2020 at 10-min intervals (5-min from Sep 2022)

Usage:
  python fetch_historical_odds.py --api-key YOUR_KEY --start 2020-10-01 --end 2025-02-01
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import pandas as pd

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"

# Priority order for bookmakers (Pinnacle is sharpest)
BOOKMAKER_PRIORITY = [
    'pinnacle',
    'bovada', 
    'betonlineag',
    'fanduel',
    'draftkings',
    'betmgm',
    'caesars',
]

DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'historical_odds_api'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_historical_odds(api_key: str, date: str, regions: str = 'us') -> dict:
    """
    Fetch historical odds for a specific date/time.
    
    Args:
        api_key: The Odds API key
        date: ISO 8601 timestamp (e.g., '2024-01-15T00:00:00Z')
        regions: Region code (us, uk, eu, au)
    
    Returns:
        API response dict with timestamp, data, and navigation
    """
    url = f"{API_BASE}/historical/sports/{SPORT}/odds"
    params = {
        'apiKey': api_key,
        'date': date,
        'regions': regions,
        'markets': 'h2h',
        'oddsFormat': 'american',
    }
    
    resp = requests.get(url, params=params, timeout=30)
    
    # Check remaining quota
    remaining = resp.headers.get('x-requests-remaining', '?')
    used = resp.headers.get('x-requests-used', '?')
    cost = resp.headers.get('x-requests-last', '?')
    
    print(f"  API: {remaining} remaining, {used} used, cost {cost}")
    
    if resp.status_code == 422:
        print(f"  ⚠️  No data for {date}")
        return None
    
    resp.raise_for_status()
    return resp.json()


def extract_closing_odds(data: dict) -> list:
    """
    Extract closing odds from API response.
    
    Returns list of dicts with:
        - game_id, commence_time, home_team, away_team
        - bookmaker, home_odds, away_odds
        - timestamp (snapshot time)
    """
    if not data or 'data' not in data:
        return []
    
    timestamp = data.get('timestamp')
    games = []
    
    for event in data['data']:
        game_id = event['id']
        commence_time = event['commence_time']
        home_team = event['home_team']
        away_team = event['away_team']
        
        # Get best available bookmaker odds
        best_odds = None
        best_bookmaker = None
        
        for bm in event.get('bookmakers', []):
            bm_key = bm['key']
            
            # Check priority
            if best_bookmaker is None or (
                bm_key in BOOKMAKER_PRIORITY and 
                (best_bookmaker not in BOOKMAKER_PRIORITY or 
                 BOOKMAKER_PRIORITY.index(bm_key) < BOOKMAKER_PRIORITY.index(best_bookmaker))
            ):
                for market in bm.get('markets', []):
                    if market['key'] == 'h2h':
                        outcomes = {o['name']: o['price'] for o in market['outcomes']}
                        if home_team in outcomes and away_team in outcomes:
                            best_odds = {
                                'home_odds': outcomes[home_team],
                                'away_odds': outcomes[away_team],
                            }
                            best_bookmaker = bm_key
                            break
        
        if best_odds:
            games.append({
                'game_id': game_id,
                'commence_time': commence_time,
                'home_team': home_team,
                'away_team': away_team,
                'bookmaker': best_bookmaker,
                'home_odds': best_odds['home_odds'],
                'away_odds': best_odds['away_odds'],
                'snapshot_time': timestamp,
            })
    
    return games


def get_nhl_game_dates(start_date: str, end_date: str) -> list:
    """
    Generate list of dates when NHL games typically occur.
    NHL season: Oct - Apr (regular), Apr - Jun (playoffs)
    Games typically at 7pm ET = midnight UTC
    """
    dates = []
    current = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
    
    while current <= end:
        month = current.month
        # NHL season months (Oct-Jun, skip Jul-Sep)
        if month >= 10 or month <= 6:
            # Request snapshot at game time (roughly midnight UTC for 7pm ET games)
            # Actually, let's request a bit before games start to get pre-game odds
            snapshot_time = current.replace(hour=23, minute=0, second=0)
            dates.append(snapshot_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
        
        current += timedelta(days=1)
    
    return dates


def fetch_season(api_key: str, season: str, output_file: Path) -> pd.DataFrame:
    """
    Fetch all odds for an NHL season.
    
    Args:
        api_key: The Odds API key
        season: Season string like '2023-2024'
        output_file: Path to save CSV
    
    Returns:
        DataFrame with all games and odds
    """
    # Parse season dates
    start_year = int(season.split('-')[0])
    start_date = f"{start_year}-10-01T00:00:00Z"
    end_date = f"{start_year + 1}-06-30T00:00:00Z"
    
    print(f"\n{'='*60}")
    print(f"Fetching {season} season: {start_date[:10]} to {end_date[:10]}")
    print(f"{'='*60}")
    
    dates = get_nhl_game_dates(start_date, end_date)
    print(f"Will query {len(dates)} dates")
    
    games_dict = {}  # game_id -> game data (keep latest/closest to game time)
    
    for i, date in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] {date[:10]}")
        
        try:
            data = fetch_historical_odds(api_key, date)
            if data:
                games = extract_closing_odds(data)
                
                # Keep latest snapshot for each game (overwrite earlier ones)
                for game in games:
                    game_id = game['game_id']
                    # Always update - later snapshots are closer to game time (closing odds)
                    games_dict[game_id] = game
                
                print(f"  Found {len(games)} games, total unique: {len(games_dict)}")
            
            # Rate limiting - be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            time.sleep(2)
    
    # Convert to DataFrame
    all_games = list(games_dict.values())
    df = pd.DataFrame(all_games)
    
    if not df.empty:
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved {len(df)} games to {output_file}")
    else:
        print(f"\n⚠️  No games found for {season}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Fetch historical NHL odds')
    parser.add_argument('--api-key', required=True, help='The Odds API key')
    parser.add_argument('--season', help='Single season to fetch (e.g., 2023-2024)')
    parser.add_argument('--all-seasons', action='store_true', help='Fetch all seasons 2020-2025')
    parser.add_argument('--test', action='store_true', help='Test with single date')
    
    args = parser.parse_args()
    
    if args.test:
        # Test single request
        print("Testing API connection...")
        data = fetch_historical_odds(args.api_key, '2024-01-15T00:00:00Z')
        if data:
            games = extract_closing_odds(data)
            print(f"Found {len(games)} games")
            for g in games[:3]:
                print(f"  {g['away_team']} @ {g['home_team']}: {g['home_odds']:+d} / {g['away_odds']:+d} ({g['bookmaker']})")
        return
    
    seasons = []
    if args.all_seasons:
        seasons = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    elif args.season:
        seasons = [args.season]
    else:
        print("Specify --season or --all-seasons")
        return
    
    all_data = []
    for season in seasons:
        output_file = DATA_DIR / f'nhl_odds_{season}.csv'
        
        # Skip if already fetched
        if output_file.exists():
            print(f"Skipping {season} - already exists")
            df = pd.read_csv(output_file)
            all_data.append(df)
            continue
        
        df = fetch_season(args.api_key, season, output_file)
        if not df.empty:
            all_data.append(df)
    
    # Combine all seasons
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_file = DATA_DIR / 'nhl_odds_all.csv'
        combined.to_csv(combined_file, index=False)
        print(f"\n{'='*60}")
        print(f"TOTAL: {len(combined)} games saved to {combined_file}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()

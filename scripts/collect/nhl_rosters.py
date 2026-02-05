#!/usr/bin/env python3
"""
NHL Official Roster Collector

Fetches current rosters from NHL API for all teams.
Provides authoritative player data including:
- Forwards, Defensemen, Goalies
- Jersey numbers, positions, height/weight
- Birth info, shooting hand

Usage:
    python nhl_rosters.py              # All teams
    python nhl_rosters.py NYR LAK VGK  # Specific teams
    python nhl_rosters.py --json       # Output JSON only
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import time

NHL_API_BASE = "https://api-web.nhle.com/v1"

# All NHL team abbreviations
ALL_TEAMS = [
    'ANA', 'ARI', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL',
    'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR',
    'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'UTA', 'VAN',
    'VGK', 'WSH', 'WPG'
]

# Map some alternate abbreviations
TEAM_ALIASES = {
    'LA': 'LAK',
    'SJ': 'SJS',
    'TB': 'TBL',
    'NJ': 'NJD',
    'WAS': 'WSH',
    'VEG': 'VGK',
}


def fetch_roster(team: str) -> dict:
    """Fetch roster for a single team from NHL API."""
    team = TEAM_ALIASES.get(team.upper(), team.upper())
    
    url = f"{NHL_API_BASE}/roster/{team}/current"
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; NHLModel/2.0)',
        'Accept': 'application/json'
    }
    
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except HTTPError as e:
        print(f"  Error fetching {team}: {e}")
        return None
    except Exception as e:
        print(f"  Error fetching {team}: {e}")
        return None


def parse_player(player: dict) -> dict:
    """Parse player data into simplified format."""
    return {
        'id': player.get('id'),
        'first_name': player.get('firstName', {}).get('default', ''),
        'last_name': player.get('lastName', {}).get('default', ''),
        'name': f"{player.get('firstName', {}).get('default', '')} {player.get('lastName', {}).get('default', '')}",
        'number': player.get('sweaterNumber'),
        'position': player.get('positionCode'),
        'shoots': player.get('shootsCatches'),
        'height_in': player.get('heightInInches'),
        'weight_lb': player.get('weightInPounds'),
        'birth_date': player.get('birthDate'),
        'birth_country': player.get('birthCountry'),
    }


def parse_roster(data: dict, team: str) -> dict:
    """Parse full roster data."""
    if not data:
        return {'error': 'No data'}
    
    forwards = [parse_player(p) for p in data.get('forwards', [])]
    defensemen = [parse_player(p) for p in data.get('defensemen', [])]
    goalies = [parse_player(p) for p in data.get('goalies', [])]
    
    return {
        'team': team,
        'forwards': forwards,
        'defensemen': defensemen,
        'goalies': goalies,
        'total_players': len(forwards) + len(defensemen) + len(goalies),
        'fetched_at': datetime.now(timezone.utc).isoformat()
    }


def get_all_rosters(teams: list = None, delay: float = 0.1) -> dict:
    """Fetch rosters for all or specified teams."""
    teams = teams or ALL_TEAMS
    rosters = {}
    
    for team in teams:
        team = TEAM_ALIASES.get(team.upper(), team.upper())
        print(f"  Fetching {team}...", end=' ', flush=True)
        
        data = fetch_roster(team)
        roster = parse_roster(data, team)
        
        if 'error' not in roster:
            print(f"✓ {roster['total_players']} players")
        else:
            print(f"✗ {roster.get('error', 'Failed')}")
        
        rosters[team] = roster
        time.sleep(delay)  # Rate limiting
    
    return rosters


def compare_rosters(old_rosters: dict, new_rosters: dict) -> list:
    """Compare rosters to find changes (trades, call-ups, etc.)."""
    changes = []
    
    for team, new_roster in new_rosters.items():
        if 'error' in new_roster:
            continue
        
        old_roster = old_rosters.get(team, {})
        if 'error' in old_roster or not old_roster:
            continue
        
        # Get player IDs
        old_ids = set()
        new_ids = set()
        old_players = {}
        new_players = {}
        
        for pos in ['forwards', 'defensemen', 'goalies']:
            for p in old_roster.get(pos, []):
                old_ids.add(p['id'])
                old_players[p['id']] = p
            for p in new_roster.get(pos, []):
                new_ids.add(p['id'])
                new_players[p['id']] = p
        
        # Find additions
        for pid in new_ids - old_ids:
            p = new_players[pid]
            changes.append({
                'type': 'ADDED',
                'team': team,
                'player': p['name'],
                'position': p['position']
            })
        
        # Find removals
        for pid in old_ids - new_ids:
            p = old_players[pid]
            changes.append({
                'type': 'REMOVED',
                'team': team,
                'player': p['name'],
                'position': p['position']
            })
    
    return changes


def main():
    parser = argparse.ArgumentParser(description='Fetch NHL rosters')
    parser.add_argument('teams', nargs='*', help='Teams to fetch (default: all)')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    parser.add_argument('--compare', action='store_true', help='Compare with previous fetch')
    args = parser.parse_args()
    
    teams = [t.upper() for t in args.teams] if args.teams else ALL_TEAMS
    
    if not args.json:
        print(f"Fetching NHL rosters for {len(teams)} teams...")
    
    rosters = get_all_rosters(teams)
    
    # Save to file
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'rosters'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'nhl_rosters.json'
    
    # Load old rosters for comparison if requested
    old_rosters = {}
    if args.compare and output_file.exists():
        with open(output_file, 'r') as f:
            old_data = json.load(f)
            old_rosters = old_data.get('teams', {})
    
    output_data = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'team_count': len(teams),
        'teams': rosters
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    if args.json:
        print(json.dumps(output_data, indent=2))
        return rosters
    
    # Summary
    total_players = sum(r.get('total_players', 0) for r in rosters.values() if 'error' not in r)
    print(f"\n✓ Fetched {total_players} players across {len(teams)} teams")
    print(f"✓ Saved to {output_file}")
    
    # Show changes if comparing
    if args.compare and old_rosters:
        changes = compare_rosters(old_rosters, rosters)
        if changes:
            print(f"\n{'='*50}")
            print("ROSTER CHANGES DETECTED")
            print('='*50)
            for c in changes:
                emoji = '➕' if c['type'] == 'ADDED' else '➖'
                print(f"  {emoji} {c['team']}: {c['player']} ({c['position']})")
    
    return rosters


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Daily Faceoff Roster & Injury Collector (V2 - JSON extraction)

Scrapes Daily Faceoff __NEXT_DATA__ for:
- Player rosters with positions
- Injury reports (IR, DTD, OUT, GTD)
- Goalie availability

Usage:
    python daily_faceoff_rosters.py          # All teams
    python daily_faceoff_rosters.py NYR CAR  # Specific teams
"""

import json
import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# Team abbreviations to Daily Faceoff URL slugs
TEAM_SLUGS = {
    'ANA': 'anaheim-ducks', 'ARI': 'utah-hockey-club', 'BOS': 'boston-bruins',
    'BUF': 'buffalo-sabres', 'CGY': 'calgary-flames', 'CAR': 'carolina-hurricanes',
    'CHI': 'chicago-blackhawks', 'COL': 'colorado-avalanche', 'CBJ': 'columbus-blue-jackets',
    'DAL': 'dallas-stars', 'DET': 'detroit-red-wings', 'EDM': 'edmonton-oilers',
    'FLA': 'florida-panthers', 'LAK': 'los-angeles-kings', 'MIN': 'minnesota-wild',
    'MTL': 'montreal-canadiens', 'NSH': 'nashville-predators', 'NJD': 'new-jersey-devils',
    'NYI': 'new-york-islanders', 'NYR': 'new-york-rangers', 'OTT': 'ottawa-senators',
    'PHI': 'philadelphia-flyers', 'PIT': 'pittsburgh-penguins', 'SJS': 'san-jose-sharks',
    'SEA': 'seattle-kraken', 'STL': 'st-louis-blues', 'TBL': 'tampa-bay-lightning',
    'TOR': 'toronto-maple-leafs', 'UTA': 'utah-hockey-club', 'VAN': 'vancouver-canucks',
    'VGK': 'vegas-golden-knights', 'WSH': 'washington-capitals', 'WPG': 'winnipeg-jets'
}

# Known goalie player IDs (will be expanded dynamically)
KNOWN_GOALIES = {
    25267: 'Igor Shesterkin',
    771: 'Jonathan Quick',
    2721: 'Spencer Martin',
    # Add more as discovered
}


def fetch_page(url: str) -> str:
    """Fetch a page with appropriate headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; NHLModel/2.0)',
        'Accept': 'text/html'
    }
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8')
    except HTTPError as e:
        print(f"Error fetching {url}: {e}")
        return ""


def extract_next_data(html: str) -> dict:
    """Extract __NEXT_DATA__ JSON from page."""
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    return {}


def get_team_roster(team: str) -> dict:
    """Get roster info for a team including injuries from JSON data."""
    slug = TEAM_SLUGS.get(team.upper())
    if not slug:
        return {'error': f'Unknown team: {team}'}
    
    url = f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations/"
    html = fetch_page(url)
    
    if not html:
        return {'error': f'Failed to fetch roster for {team}'}
    
    # Extract JSON data
    data = extract_next_data(html)
    if not data:
        return {'error': f'No JSON data found for {team}'}
    
    props = data.get('props', {}).get('pageProps', {})
    combinations = props.get('combinations', {})
    players = combinations.get('players', [])
    
    # Get last updated time
    last_updated = combinations.get('updatedAt')
    
    # Extract injuries
    injuries = []
    for p in players:
        status = p.get('injuryStatus')
        if status:
            injuries.append({
                'name': p.get('name'),
                'player_id': p.get('playerId'),
                'status': status.upper(),
                'position': p.get('positionName') or p.get('positionIdentifier'),
            })
    
    # Extract goalies (from position or known IDs)
    goalies = []
    for p in players:
        pos = p.get('positionIdentifier') or p.get('positionName') or ''
        is_goalie = (
            'G' in pos.upper() or 
            'goalie' in pos.lower() or
            p.get('playerId') in KNOWN_GOALIES
        )
        if is_goalie or (p.get('playerId') in KNOWN_GOALIES):
            injured = p.get('injuryStatus') is not None
            goalies.append({
                'name': p.get('name'),
                'player_id': p.get('playerId'),
                'jersey': p.get('jerseyNumber'),
                'injured': injured,
                'injury_status': p.get('injuryStatus', '').upper() if injured else None
            })
    
    # Also check if any injured players are goalies (by known IDs or name patterns)
    for inj in injuries:
        pid = inj.get('player_id')
        name = inj.get('name', '').lower()
        # Check if this injured player is a known goalie
        if pid in KNOWN_GOALIES or 'shesterkin' in name or 'quick' in name:
            # Add to goalies if not already there
            if not any(g.get('player_id') == pid for g in goalies):
                goalies.append({
                    'name': inj.get('name'),
                    'player_id': pid,
                    'jersey': None,
                    'injured': True,
                    'injury_status': inj.get('status')
                })
    
    return {
        'team': team.upper(),
        'last_updated': last_updated,
        'goalies': goalies,
        'injuries': injuries,
        'total_players': len(players),
        'fetched_at': datetime.now(timezone.utc).isoformat()
    }


def main():
    teams = sys.argv[1:] if len(sys.argv) > 1 else list(TEAM_SLUGS.keys())
    
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'rosters'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_rosters = {}
    critical_injuries = []
    
    for team in teams:
        team = team.upper()
        print(f"Fetching {team}...", end=' ', flush=True)
        roster = get_team_roster(team)
        
        if 'error' not in roster:
            injuries = roster.get('injuries', [])
            goalies = roster.get('goalies', [])
            injured_goalies = [g for g in goalies if g.get('injured')]
            
            print(f"✓ {len(injuries)} injuries", end='')
            if injured_goalies:
                print(f", {len(injured_goalies)} goalie(s) OUT")
                for g in injured_goalies:
                    status = g.get('injury_status', 'OUT')
                    print(f"  ⚠️  {team} GOALIE: {g['name']} ({status})")
                    critical_injuries.append({
                        'team': team,
                        'player': g['name'],
                        'position': 'G',
                        'status': status
                    })
            else:
                print()
        else:
            print(f"✗ {roster['error']}")
        
        all_rosters[team] = roster
    
    # Save to file
    output_file = output_dir / 'daily_faceoff_rosters.json'
    with open(output_file, 'w') as f:
        json.dump({
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'teams': all_rosters,
            'critical_injuries': critical_injuries
        }, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")
    
    # Summary of critical injuries
    if critical_injuries:
        print("\n" + "="*50)
        print("⚠️  CRITICAL INJURIES (Goalies)")
        print("="*50)
        for ci in critical_injuries:
            print(f"  {ci['team']}: {ci['player']} ({ci['status']})")
    else:
        print("\n✓ No critical goalie injuries found")
    
    return critical_injuries


if __name__ == '__main__':
    main()

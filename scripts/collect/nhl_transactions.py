#!/usr/bin/env python3
"""
NHL Transaction Scraper

Scrapes ESPN NHL Transactions page for recent moves:
- Trades
- Signings  
- IR placements
- Recalls/sends to AHL

Usage:
    python nhl_transactions.py              # Last 3 days
    python nhl_transactions.py --days 7     # Last 7 days
    python nhl_transactions.py --teams NYR LAK  # Filter to specific teams
"""

import json
import re
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

ESPN_URL = "https://www.espn.com/nhl/transactions"

# Team name to abbreviation (for trade partner extraction)
NAME_TO_ABBREV = {
    'anaheim': 'ANA', 'arizona': 'ARI', 'boston': 'BOS', 'buffalo': 'BUF',
    'calgary': 'CGY', 'carolina': 'CAR', 'chicago': 'CHI', 'colorado': 'COL',
    'columbus': 'CBJ', 'dallas': 'DAL', 'detroit': 'DET', 'edmonton': 'EDM',
    'florida': 'FLA', 'los angeles': 'LAK', 'la': 'LAK', 'kings': 'LAK',
    'minnesota': 'MIN', 'montreal': 'MTL', 'nashville': 'NSH', 
    'new jersey': 'NJD', 'devils': 'NJD',
    'new york islanders': 'NYI', 'islanders': 'NYI',
    'new york rangers': 'NYR', 'rangers': 'NYR',
    'ottawa': 'OTT', 'philadelphia': 'PHI', 'pittsburgh': 'PIT', 
    'san jose': 'SJS', 'seattle': 'SEA', 
    'st. louis': 'STL', 'st louis': 'STL', 'blues': 'STL',
    'tampa bay': 'TBL', 'lightning': 'TBL',
    'toronto': 'TOR', 'maple leafs': 'TOR',
    'utah': 'UTA', 'vancouver': 'VAN', 'vegas': 'VGK', 'golden knights': 'VGK',
    'washington': 'WSH', 'capitals': 'WSH', 'winnipeg': 'WPG', 'jets': 'WPG',
}


def fetch_transactions() -> list:
    """Fetch and parse ESPN transactions from embedded JSON."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html'
    }
    req = Request(ESPN_URL, headers=headers)
    
    try:
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
    except HTTPError as e:
        print(f"Error fetching ESPN: {e}")
        return []
    
    # Extract transactions from embedded JSON
    # Pattern: {"date":"...","description":"...","team":{..."abbreviation":"XXX"...}}
    pattern = re.compile(
        r'\{"date":"([^"]+)","description":"([^"]+)","team":\{[^}]*"abbreviation":"([^"]+)"',
        re.DOTALL
    )
    
    transactions = []
    for match in pattern.finditer(html):
        date_str, description, team = match.groups()
        
        # Parse date
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            date = dt.date().isoformat()
        except:
            date = datetime.now().date().isoformat()
        
        trans = {
            'date': date,
            'team': team,
            'text': description,
            'type': classify_transaction(description),
            'players': extract_players(description),
        }
        
        # Extract trade partner
        if trans['type'] == 'TRADE':
            partner = extract_trade_partner(description)
            if partner:
                trans['trade_partner'] = partner
        
        transactions.append(trans)
    
    return transactions


def classify_transaction(text: str) -> str:
    """Classify transaction type."""
    text_lower = text.lower()
    if 'acquired' in text_lower or 'traded' in text_lower or 'in exchange for' in text_lower:
        return 'TRADE'
    elif 'injured reserve' in text_lower:
        return 'IR'
    elif 'signed' in text_lower and 'contract' in text_lower:
        return 'SIGNING'
    elif 'recalled' in text_lower or 'sent' in text_lower:
        return 'ROSTER'
    elif 'waiver' in text_lower:
        return 'WAIVER'
    elif 'retire' in text_lower:
        return 'RETIREMENT'
    else:
        return 'OTHER'


def extract_players(text: str) -> list:
    """Extract player names from transaction text."""
    pattern = re.compile(r'\b([FGDCL]W?)\s+([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)')
    matches = pattern.findall(text)
    return [{'position': m[0], 'name': m[1]} for m in matches]


def extract_trade_partner(text: str) -> str:
    """Extract trade partner team from transaction text."""
    text_lower = text.lower()
    
    # Patterns: "from the New York Rangers", "to St. Louis"
    patterns = [
        r'from\s+(?:the\s+)?([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*)',
        r'to\s+(?:the\s+)?([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)*)\s+(?:in exchange|for)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            team_name = match.group(1).lower().strip()
            # Try exact match first
            if team_name in NAME_TO_ABBREV:
                return NAME_TO_ABBREV[team_name]
            # Try partial matches
            for name, abbrev in NAME_TO_ABBREV.items():
                if name in team_name or team_name in name:
                    return abbrev
    
    return None


def filter_transactions(transactions: list, days: int = 3, teams: list = None) -> list:
    """Filter transactions by recency and teams."""
    cutoff = datetime.now().date() - timedelta(days=days)
    
    filtered = []
    for t in transactions:
        # Filter by date
        try:
            trans_date = datetime.fromisoformat(t['date']).date()
            if trans_date < cutoff:
                continue
        except:
            pass
        
        # Filter by team
        if teams:
            team_match = t.get('team') in teams
            partner_match = t.get('trade_partner') in teams
            if not team_match and not partner_match:
                continue
        
        filtered.append(t)
    
    return filtered


def get_todays_impact(transactions: list, teams: list) -> list:
    """Get high-impact transactions for today's games."""
    today = datetime.now().date().isoformat()
    yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
    
    impact = []
    for t in transactions:
        if t['date'] not in [today, yesterday]:
            continue
        
        team_involved = t.get('team') in teams or t.get('trade_partner') in teams
        if not team_involved:
            continue
        
        # Trades are high impact
        if t['type'] == 'TRADE':
            impact.append({**t, 'impact': 'HIGH', 'reason': 'Trade affects lineup'})
        # IR for goalies
        elif t['type'] == 'IR':
            for p in t.get('players', []):
                if p['position'] == 'G':
                    impact.append({**t, 'impact': 'HIGH', 'reason': f"Goalie {p['name']} to IR"})
                    break
    
    return impact


def main():
    parser = argparse.ArgumentParser(description='Fetch NHL transactions')
    parser.add_argument('--days', type=int, default=3, help='Days of history')
    parser.add_argument('--teams', nargs='*', help='Filter to specific teams')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    args = parser.parse_args()
    
    if not args.json:
        print(f"Fetching NHL transactions (last {args.days} days)...")
    
    transactions = fetch_transactions()
    
    if not args.json:
        print(f"Found {len(transactions)} total transactions")
    
    # Filter
    teams = [t.upper() for t in args.teams] if args.teams else None
    filtered = filter_transactions(transactions, days=args.days, teams=teams)
    
    # Save to file
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'transactions'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'recent_transactions.json'
    output_data = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'days': args.days,
        'teams_filter': teams,
        'count': len(filtered),
        'transactions': filtered
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    if args.json:
        print(json.dumps(output_data, indent=2))
        return filtered
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TRANSACTIONS (Last {args.days} days)")
    if teams:
        print(f"Filtered to: {', '.join(teams)}")
    print('='*60)
    
    if not filtered:
        print("\nNo transactions found matching filters.")
        return filtered
    
    # Group by date
    by_date = {}
    for t in filtered:
        d = t.get('date', 'Unknown')
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(t)
    
    for date in sorted(by_date.keys(), reverse=True):
        print(f"\n📅 {date}")
        for t in by_date[date]:
            type_emoji = {
                'TRADE': '🔄',
                'IR': '🏥',
                'SIGNING': '✍️',
                'ROSTER': '📋',
                'WAIVER': '📝',
                'RETIREMENT': '👋',
                'OTHER': '📌'
            }.get(t['type'], '•')
            
            text = t['text'][:70] + '...' if len(t['text']) > 70 else t['text']
            print(f"  {type_emoji} {t['team']}: {text}")
            
            if t.get('trade_partner'):
                print(f"      ↔️  Trade with: {t['trade_partner']}")
    
    # Highlight high-impact for today's teams
    if teams:
        impact = get_todays_impact(filtered, teams)
        if impact:
            print(f"\n{'='*60}")
            print("⚠️  HIGH IMPACT FOR TODAY'S GAMES")
            print('='*60)
            for t in impact:
                print(f"  🚨 {t['team']}: {t['text'][:60]}...")
                print(f"     Reason: {t['reason']}")
    
    print(f"\n✓ Saved to {output_file}")
    
    return filtered


if __name__ == '__main__':
    main()

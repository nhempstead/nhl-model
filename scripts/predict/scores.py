#!/usr/bin/env python3
"""
Check live NHL scores and update results.
"""

import requests
from datetime import date, datetime
import os

def get_live_scores():
    """Fetch current NHL scores"""
    today = date.today().strftime('%Y-%m-%d')
    url = f"https://api-web.nhle.com/v1/score/{today}"
    
    resp = requests.get(url, timeout=10)
    data = resp.json()
    
    games = []
    for g in data.get('games', []):
        if g.get('gameType') == 2:  # Regular season
            game = {
                'game_id': g['id'],
                'home_team': g['homeTeam']['abbrev'],
                'away_team': g['awayTeam']['abbrev'],
                'home_score': g['homeTeam'].get('score', 0),
                'away_score': g['awayTeam'].get('score', 0),
                'state': g.get('gameState'),
                'period': g.get('period', 0),
                'clock': g.get('clock', {}).get('timeRemaining', ''),
                'start_time': g.get('startTimeUTC'),
            }
            games.append(game)
    
    return games


def print_scores():
    """Print formatted scores"""
    games = get_live_scores()
    
    print(f"NHL SCORES - {date.today()}")
    print("=" * 50)
    
    for g in games:
        status = ""
        if g['state'] == 'LIVE':
            status = f"P{g['period']} {g['clock']}"
        elif g['state'] == 'FINAL':
            status = "FINAL"
        elif g['state'] == 'FUT':
            # Parse start time
            try:
                start = datetime.fromisoformat(g['start_time'].replace('Z', '+00:00'))
                status = start.strftime('%I:%M %p')
            except:
                status = "Scheduled"
        else:
            status = g['state']
        
        print(f"{g['away_team']} {g['away_score']} @ {g['home_team']} {g['home_score']}  [{status}]")
    
    return games


def check_picks_results(picks):
    """Check results for our picks"""
    games = get_live_scores()
    
    results = []
    for pick in picks:
        # Find matching game
        for g in games:
            if pick['home_team'] == g['home_team'] or pick['away_team'] == g['away_team']:
                if g['state'] == 'FINAL':
                    winner = g['home_team'] if g['home_score'] > g['away_score'] else g['away_team']
                    won = pick['pick'] == winner
                    results.append({
                        **pick,
                        'result': 'W' if won else 'L',
                        'final_score': f"{g['away_team']} {g['away_score']} - {g['home_team']} {g['home_score']}"
                    })
                break
    
    return results


if __name__ == '__main__':
    print_scores()
